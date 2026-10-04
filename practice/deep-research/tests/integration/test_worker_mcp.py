import contextvars
import json
import threading
import time

from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.supervisor import ResearchSupervisor
from intelligence_collector.agents.worker import run_worker
from intelligence_collector.contracts import ResearchTodo, TodoStatus
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.fake_models import FakeResearchModel


def _researcher(model, mcp_factory, **kwargs):
    return ResearchAgent(ModelRouter(model, **kwargs), mcp_factory)


def test_research_agent_searches_broad_then_narrow_and_persists_provenance(stores):
    vfs, index = stores
    todo = ResearchTodo(research_question="Compare database transaction behavior")
    narrow = "database locking behavior primary documentation"
    model = FakeResearchModel(queries={todo.research_question: [todo.research_question, narrow]})
    mcp = ControlledMCP(
        [
            [{"url": "https://primary.test/overview", "content": "Transaction overview"}],
            [{"url": "https://primary.test/locking", "content": "Documented locking behavior"}],
        ]
    )
    traces = []

    completed, refs, chunks = run_worker(
        vfs,
        index,
        "run",
        todo,
        _researcher(model, lambda: mcp, trace=lambda **event: traces.append(event)),
    )

    assert completed.status == TodoStatus.COMPLETED
    assert [request["query"] for request in mcp.requests] == [todo.research_question, narrow]
    assert model.bound_tools[0]["function"]["name"] == "tavily_search"
    # Only the first turn forces a search; later turns may finish without one.
    assert model.tool_choices == ["any", None, None]
    assert [event["role"] for event in traces] == [
        "researcher",
        "researcher",
        "researcher",
        "research_compressor",
    ]
    assert {ref.kind for ref in refs} == {
        "raw_capture",
        "url_registry",
        "notes",
        "excerpts",
        "evidence_bundle",
    }
    evidence = json.loads(vfs.read("run", f"research/{todo.todo_id}/evidence/result.json"))
    assert evidence["sufficient"] is True
    assert evidence["search_calls"] == 2
    assert evidence["summary"].startswith("Findings for")
    assert evidence["sources"][0]["mcp_server"] == "controlled"
    assert evidence["sources"][0]["provenance"]["query"] == todo.research_question
    assert evidence["sources"][1]["provenance"]["query"] == narrow
    assert evidence["sources"][0]["content_hash"]
    notes = vfs.read("run", f"research/{todo.todo_id}/notes/result.md").decode()
    assert "## Findings" in notes and "Search calls: 2/5" in notes
    assert all(chunk.metadata["locator"].startswith("https://") for chunk in chunks)


def test_research_agent_stops_when_llm_gives_final_answer(stores):
    vfs, index = stores
    todo = ResearchTodo(research_question="Find the transaction specification")
    mcp = ControlledMCP([{"url": "https://primary.test/spec", "content": "Complete specification"}])

    completed, _, _ = run_worker(
        vfs, index, "early", todo, _researcher(FakeResearchModel(), lambda: mcp)
    )

    assert completed.status == TodoStatus.COMPLETED
    assert mcp.calls == 1


def test_search_budget_is_a_hard_per_task_limit_and_gaps_are_recorded(stores):
    vfs, index = stores
    todo = ResearchTodo(research_question="Investigate an unresolved compatibility issue")
    model = FakeResearchModel(
        queries={todo.research_question: [f"query {n}" for n in range(8)]},
        findings={
            todo.research_question: {
                "summary": "Partial evidence [https://primary.test/partial]",
                "sufficient": False,
                "unresolved_gaps": ["cross-platform confirmation"],
            }
        },
    )
    mcp = ControlledMCP([{"url": "https://primary.test/partial", "content": "Partial evidence"}])

    completed, _, _ = run_worker(vfs, index, "limited", todo, _researcher(model, lambda: mcp))

    evidence = json.loads(vfs.read("limited", f"research/{todo.todo_id}/evidence/result.json"))
    assert mcp.calls == 5
    assert completed.status == TodoStatus.COMPLETED  # best available outcome, gap recorded
    assert evidence["sufficient"] is False
    assert evidence["unresolved_gaps"] == ["cross-platform confirmation"]
    assert evidence["search_calls"] == 5


def test_source_budget_caps_distinct_pages_per_todo_across_all_searches(stores):
    """One todo whose searches each return 5 new pages (the case that previously kept 17 pages
    over 4 searches) keeps at most 5 in total, asks each search only for the slots left, and
    stops searching once the budget is full."""
    vfs, index = stores
    todo = ResearchTodo(research_question="What is the latest version of TypeScript?")
    model = FakeResearchModel(queries={todo.research_question: [f"query {n}" for n in range(4)]})
    pages = [
        [
            {"url": f"https://page.test/{batch}-{n}", "content": f"page {batch}-{n}"}
            for n in range(5)
        ]
        for batch in range(4)
    ]
    mcp = ControlledMCP(pages)
    requested_limits = []
    original_search = mcp.search

    def recording_search(query, **kwargs):
        requested_limits.append(kwargs.get("max_results"))
        return original_search(query, **kwargs)

    mcp.search = recording_search

    completed, _, _ = run_worker(vfs, index, "budget", todo, _researcher(model, lambda: mcp))

    evidence = json.loads(vfs.read("budget", f"research/{todo.todo_id}/evidence/result.json"))
    assert completed.status == TodoStatus.COMPLETED
    assert len(evidence["sources"]) == 5
    assert requested_limits == [5]  # first search filled the budget; no further searches ran
    assert mcp.calls == 1


def test_source_budget_requests_only_the_remaining_slots(stores):
    vfs, index = stores
    todo = ResearchTodo(research_question="Compare two release notes")
    model = FakeResearchModel(queries={todo.research_question: ["broad", "gap"]})
    mcp = ControlledMCP(
        [
            [{"url": f"https://first.test/{n}", "content": "first"} for n in range(3)],
            [{"url": f"https://second.test/{n}", "content": "second"} for n in range(5)],
        ]
    )
    requested_limits = []
    original_search = mcp.search

    def recording_search(query, **kwargs):
        requested_limits.append(kwargs.get("max_results"))
        return original_search(query, **kwargs)

    mcp.search = recording_search

    run_worker(vfs, index, "remaining", todo, _researcher(model, lambda: mcp))

    evidence = json.loads(vfs.read("remaining", f"research/{todo.todo_id}/evidence/result.json"))
    assert requested_limits == [5, 2]
    assert [s["locator"] for s in evidence["sources"]] == [
        "https://first.test/0",
        "https://first.test/1",
        "https://first.test/2",
        "https://second.test/0",
        "https://second.test/1",
    ]


def test_mcp_failure_is_persisted_as_task_outcome_without_leaking_key(stores):
    class FailingMCP:
        max_calls = 5

        def search(self, query, **kwargs):
            raise TimeoutError("GET https://mcp.test/?tavilyApiKey=super-secret timed out")

    vfs, index = stores
    todo = ResearchTodo(research_question="Find primary documentation")

    completed, refs, chunks = run_worker(
        vfs, index, "failed", todo, _researcher(FakeResearchModel(), FailingMCP)
    )

    outcome_text = vfs.read("failed", f"research/{todo.todo_id}/outcome.json").decode()
    outcome = json.loads(outcome_text)
    assert completed.status == TodoStatus.FAILED
    assert [ref.kind for ref in refs] == ["research_outcome"] and chunks == []
    assert outcome["status"] == "failed"
    assert "TimeoutError" in outcome["error"]
    assert "super-secret" not in outcome_text


def test_supervisor_delegates_one_sub_agent_per_todo_in_parallel(stores):
    vfs, index = stores
    todos = [ResearchTodo(research_question=f"Question {n}") for n in range(3)]
    model = FakeResearchModel(
        queries={
            todo.research_question: [f"{todo.research_question} q{i}" for i in range(4)]
            for todo in todos
        },
    )
    clients, active, peak, lock = [], 0, 0, threading.Lock()

    class SlowMCP(ControlledMCP):
        def search(self, query, **kwargs):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return super().search(query, **kwargs)

    def factory():
        client = SlowMCP([{"url": f"https://primary.test/{len(clients)}", "content": "evidence"}])
        clients.append(client)
        return client

    supervisor = ResearchSupervisor(
        _researcher(model, factory), vfs, index, max_concurrent_researchers=2
    )
    results = supervisor.delegate("parallel", todos)

    assert [todo.todo_id for todo, _, _ in results] == [todo.todo_id for todo in todos]
    assert all(todo.status == TodoStatus.COMPLETED for todo, _, _ in results)
    assert len(clients) == 3  # a fresh MCP client, and search budget, per sub-agent
    assert all(client.calls == 4 for client in clients)
    assert peak == 2


def test_research_sub_agents_inherit_the_calling_thread_context_for_tracing(stores):
    """Sub-agents run off the event loop via asyncio.to_thread (ResearchSupervisor.
    _gather_sub_agents), which propagates the calling contextvars.Context into its worker thread
    automatically. Langfuse's OpenTelemetry-based "current span" lives in exactly that context,
    so if this ever regressed (e.g. back to a bare ThreadPoolExecutor with no context handling),
    every sub-agent's LLM call traces would detach from the run's trace instead of nesting under
    it. ModelRouter's `trace` hook fires from inside the worker thread, so it's the right place
    to observe whether the context made it."""
    vfs, index = stores
    probe: contextvars.ContextVar[str] = contextvars.ContextVar("probe")
    token = probe.set("main-thread-value")
    seen: list[str] = []

    def capture_trace(**_event):
        try:
            seen.append(probe.get())
        except LookupError:
            seen.append("<EMPTY>")

    try:
        todos = [ResearchTodo(research_question=f"Question {n}") for n in range(3)]
        supervisor = ResearchSupervisor(
            _researcher(
                FakeResearchModel(),
                lambda: ControlledMCP([{"url": "https://primary.test/0", "content": "evidence"}]),
                trace=capture_trace,
            ),
            vfs,
            index,
            max_concurrent_researchers=3,
        )

        supervisor.delegate("context-probe", todos)
    finally:
        probe.reset(token)

    assert seen  # the trace hook actually fired from inside the worker threads
    assert all(value == "main-thread-value" for value in seen)


def test_sub_agent_does_not_repeat_a_url_it_already_retrieved(stores):
    vfs, index = stores
    todo = ResearchTodo(research_question="How do caches evict entries?")
    model = FakeResearchModel(queries={todo.research_question: ["broad cache", "narrow cache"]})
    page = {"url": "https://primary.test/cache", "content": "A cache evicts entries."}
    mcp = ControlledMCP(
        [
            [page],
            [
                {
                    **page,
                    "url": "https://PRIMARY.test/cache/#eviction",
                    "content": "Different snippet",
                },
                {
                    "url": "https://primary.test/lru",
                    "content": "LRU evicts the least recently used.",
                },
            ],
        ]
    )
    captured = []
    researcher = _researcher(model, lambda: mcp)
    original = researcher.research

    def research(question, **kwargs):
        captured.append(original(question, **kwargs))
        return captured[-1]

    researcher.research = research
    _, refs, chunks = run_worker(vfs, index, "seen", todo, researcher)

    assert [s["url"] for s in captured[0].sources] == [
        "https://primary.test/cache",
        "https://primary.test/lru",
    ]
    assert len(chunks) == 2 and [r.kind for r in refs].count("raw_capture") == 2


def test_url_found_by_several_tasks_is_captured_and_indexed_once(stores):
    vfs, index = stores
    first, second = (
        ResearchTodo(research_question="First"),
        ResearchTodo(research_question="Second"),
    )
    shared = {"url": "https://primary.test/shared", "content": "Shared evidence"}
    supervisor = ResearchSupervisor(
        _researcher(FakeResearchModel(), lambda: ControlledMCP([shared])), vfs, index
    )

    results = supervisor.delegate("shared", [first, second])
    later = supervisor.delegate("shared", [ResearchTodo(research_question="Gap follow-up")])

    assert [todo.status for todo, _, _ in [*results, *later]] == [TodoStatus.COMPLETED] * 3
    assert sum(len(chunks) for _, _, chunks in [*results, *later]) == 1
    assert len(index.retrieve("Shared evidence", limit=10, run_id="shared")) == 1
    bundles = [
        json.loads(vfs.read("shared", f"research/{todo.todo_id}/evidence/result.json"))
        for todo in (first, second, later[0][0])
    ]
    original = bundles[0]["sources"][0]
    assert "reused_from_task" not in original
    for bundle in bundles[1:]:
        reused = bundle["sources"][0]
        assert reused["reused_from_task"] == first.todo_id
        assert (reused["source_id"], reused["artifact_id"]) == (
            original["source_id"],
            original["artifact_id"],
        )
    registry = json.loads(vfs.read("shared", "research/url_registry.json"))
    assert list(registry) == ["https://primary.test/shared"]


def test_page_already_found_by_another_todo_is_linked_without_using_a_source_slot(stores):
    """Todo 2's first search returns the page todo 1 already found plus 4 new ones: the shared
    page is linked (not a new capture) and does not take a slot, so todo 2's second search still
    has one slot left to fill — ending with 5 pages of its own plus the linked one."""
    vfs, index = stores
    first, second = (
        ResearchTodo(research_question="First"),
        ResearchTodo(research_question="Second"),
    )
    shared = {"url": "https://primary.test/shared", "content": "Shared evidence"}
    own = [{"url": f"https://second.test/{n}", "content": f"own {n}"} for n in range(5)]
    clients = iter(
        [  # one client per sub-agent, in plan order (one worker)
            ControlledMCP([shared]),
            ControlledMCP([[shared, *own[:4]], [own[4]]]),
        ]
    )
    model = FakeResearchModel(queries={"Second": ["second q1", "second q2"]})
    requested = []

    def factory():
        client = next(clients)
        original = client.search

        def recording(query, **kwargs):
            requested.append((query, kwargs.get("max_results")))
            return original(query, **kwargs)

        client.search = recording
        return client

    supervisor = ResearchSupervisor(
        _researcher(model, factory),
        vfs,
        index,
        max_concurrent_researchers=1,  # deterministic: todo 1 claims the shared page first
    )
    results = supervisor.delegate("linked", [first, second])

    assert [todo.status for todo, _, _ in results] == [TodoStatus.COMPLETED] * 2
    # The linked page did not use a slot: after 4 own pages, the next search asks for 1 more.
    assert requested[1:] == [("second q1", 5), ("second q2", 1)]
    second_bundle = json.loads(
        vfs.read("linked", f"research/{second.todo_id}/evidence/result.json")
    )
    locators = [s["locator"] for s in second_bundle["sources"]]
    assert locators[0] == "https://primary.test/shared"
    assert second_bundle["sources"][0]["reused_from_task"] == first.todo_id
    assert locators[1:] == [f"https://second.test/{n}" for n in range(5)]  # full 5 own slots
    assert all("_covered_by" not in s["provenance"] for s in second_bundle["sources"])
    registry = json.loads(vfs.read("linked", "research/url_registry.json"))
    assert list(registry).count("https://primary.test/shared") == 1


def test_failing_sub_agent_does_not_cancel_siblings(stores):
    vfs, index = stores
    ok, broken = ResearchTodo(research_question="Works"), ResearchTodo(research_question="Breaks")
    model = FakeResearchModel(fail_questions={"Breaks"})
    supervisor = ResearchSupervisor(
        _researcher(model, lambda: ControlledMCP([{"url": "https://ok.test", "content": "x"}])),
        vfs,
        index,
    )

    results = {
        todo.research_question: todo.status
        for todo, _, _ in supervisor.delegate("mixed", [ok, broken])
    }

    assert results == {"Works": TodoStatus.COMPLETED, "Breaks": TodoStatus.FAILED}
    outcome = json.loads(vfs.read("mixed", f"research/{broken.todo_id}/outcome.json"))
    assert "Research sub-agent failed" in outcome["error"]
