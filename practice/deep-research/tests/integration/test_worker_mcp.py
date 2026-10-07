import contextvars
import json
import threading
import time

from intelligence_collector.agents.researcher import DEFAULT_MAX_SOURCES, ResearchAgent
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
    assert [tool["function"]["name"] for tool in model.bound_tools] == [
        "tavily_search",
        "think_tool",
    ]
    # Only the first turn forces a search (not a reflection); later turns may finish without one.
    assert model.tool_choices == ["tavily_search", None, None]
    # Phrasings for the lookups first; every search's results are evaluated before the
    # searcher's next decision; the short report last.
    assert [event["role"] for event in traces] == [
        "query_rewriter",
        "researcher",
        "result_evaluator",
        "researcher",
        "result_evaluator",
        "researcher",
        "searcher_report",
    ]
    assert {ref.kind for ref in refs} == {
        "result_evaluations",
        "passage_lookups",
        "searcher_report",
        "raw_capture",
        "url_registry",
        "notes",
        "excerpts",
        "evidence_bundle",
    }
    evidence = json.loads(vfs.read("run", f"research/{todo.todo_id}/evidence/result.json"))
    assert evidence["sufficient"] is True
    assert evidence["search_calls"] == 2
    assert evidence["summary"].startswith("- Finding from src_")
    assert evidence["sources"][0]["mcp_server"] == "controlled"
    assert evidence["sources"][0]["provenance"]["query"] == todo.research_question
    assert evidence["sources"][1]["provenance"]["query"] == narrow
    assert evidence["sources"][0]["content_hash"]
    notes = vfs.read("run", f"research/{todo.todo_id}/notes/result.md").decode()
    assert "## Findings" in notes and "Search calls: 2/5" in notes
    report = json.loads(vfs.read("run", f"research/{todo.todo_id}/report.json"))
    assert report["points"][0]["status"] == "answered"
    assert report["stop_reason"] == "model_finished" and report["report_status"] == "complete"
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
    # Each search finds one new page, so the searcher keeps making progress until its budget.
    mcp = ControlledMCP(
        [[{"url": f"https://primary.test/partial-{n}", "content": "Partial evidence"}] for n in range(8)]
    )

    completed, _, _ = run_worker(vfs, index, "limited", todo, _researcher(model, lambda: mcp))

    evidence = json.loads(vfs.read("limited", f"research/{todo.todo_id}/evidence/result.json"))
    assert mcp.calls == 5
    assert completed.status == TodoStatus.COMPLETED  # best available outcome, gap recorded
    # A point the report leaves missing is the task's unresolved gap.
    assert evidence["sufficient"] is False
    assert evidence["unresolved_gaps"] == [todo.research_question]
    assert evidence["search_calls"] == 5


def test_source_budget_caps_distinct_pages_per_todo_across_all_searches(stores):
    """One todo whose searches each return 5 new pages (the case that previously kept 17 pages
    over 4 searches) keeps at most the source budget in total, asks each search only for the slots
    left, and stops searching once the budget is full."""
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
    assert len(evidence["sources"]) == DEFAULT_MAX_SOURCES
    # The first search filled the budget; no further searches ran.
    assert requested_limits == [DEFAULT_MAX_SOURCES]
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

    researcher = ResearchAgent(ModelRouter(model), lambda: mcp, max_sources=5)
    run_worker(vfs, index, "remaining", todo, researcher)

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

    queries = [todo.research_question, "primary documentation site", "official reference"]
    model = FakeResearchModel(queries={todo.research_question: queries})
    completed, refs, chunks = run_worker(
        vfs, index, "failed", todo, _researcher(model, FailingMCP)
    )

    outcome_text = vfs.read("failed", f"research/{todo.todo_id}/outcome.json").decode()
    outcome = json.loads(outcome_text)
    assert completed.status == TodoStatus.FAILED
    assert [ref.kind for ref in refs] == [
        "result_evaluations",
        "passage_lookups",
        "searcher_report",
        "research_outcome",
    ]
    assert chunks == []
    assert outcome["status"] == "failed"
    assert outcome["failure_kind"] == "technical"
    assert "TimeoutError" in outcome["error"]
    assert "super-secret" not in outcome_text
    # Two failed searches in a row end the search (FR-007); the record keeps no credential.
    record_text = vfs.read("failed", f"research/{todo.todo_id}/evaluations.json").decode()
    record = json.loads(record_text)
    assert record["stop_reason"] == "search_failures" and len(record["queries"]) == 2
    assert record["failure_kind"] == "technical"
    assert "super-secret" not in record_text


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
        client = SlowMCP(
            [
                [{"url": f"https://primary.test/{len(clients)}/{i}", "content": "evidence"}]
                for i in range(4)
            ]
        )
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
    original = researcher.search

    def search(question, **kwargs):
        captured.append(original(question, **kwargs))
        return captured[-1]

    researcher.search = search
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
        _researcher(FakeResearchModel(), lambda: ControlledMCP([shared])),
        vfs,
        index,
        max_concurrent_researchers=1,  # deterministic: the first task captures the page
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
        ResearchAgent(ModelRouter(model), factory, max_sources=5),
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


class RejectingModel(FakeResearchModel):
    """FakeResearchModel whose result evaluator rejects every row whose URL contains "spam"."""

    def invoke(self, messages, config=None):
        import re

        from langchain_core.messages import AIMessage

        if str(messages[0].content).startswith("You evaluate web search results"):
            rows = re.findall(r"^\[(\d+)\] .*\nURL: (\S+)", str(messages[-1].content), re.M)
            return AIMessage(content=json.dumps({"rows": [
                {"index": int(n), "decision": "rejected", "reason": "off_topic"}
                if "spam" in url
                else {"index": int(n), "decision": "accepted", "covers": []}
                for n, url in rows
            ]}))
        return super().invoke(messages, config)


def test_rejected_results_are_never_captured_claimed_or_indexed(stores):
    from intelligence_collector.tools.url_tracking import UrlClaims

    vfs, index = stores
    todo = ResearchTodo(todo_id="t-keep", research_question="SQLite write locking")
    long_page = "\n\n".join(f"Paragraph {n} about SQLite locking. " * 20 for n in range(12))
    mcp = ControlledMCP(
        [
            {"url": "https://docs.test/locking", "content": "SQLite locking", "raw_content": long_page},
            {"url": "https://spam.test/ads", "content": "Buy now", "raw_content": "Buy now " * 200},
        ]
    )
    claims = UrlClaims()
    researcher = _researcher(RejectingModel(), lambda: mcp)

    from intelligence_collector.agents.search_task import SearchTask

    task = SearchTask(vfs, index, researcher, "run", todo, todo.research_question, {"t-keep"})
    completed, refs, chunks = task.run(claims)

    assert completed.status == TodoStatus.COMPLETED
    assert claims.owner("https://spam.test/ads") is None
    assert claims.owner("https://docs.test/locking") == "t-keep"
    assert {chunk.metadata["locator"] for chunk in chunks} == {"https://docs.test/locking"}
    # Every chunk of the cleaned page is indexed, not only the passages selected for the searcher.
    from intelligence_collector.storage.evidence_index import chunk_texts
    from intelligence_collector.tools.page_content import clean_page

    page, _ = clean_page(long_page)
    assert len(chunks) == len(chunk_texts("", page)) > 3
    assert len(index.documents("run")) == len(chunks)
    record = json.loads(vfs.read("run", f"research/{todo.todo_id}/evaluations.json"))
    decisions = {e["url"]: (e["decision"], e["reason"]) for e in record["evaluations"]}
    assert decisions == {
        "https://docs.test/locking": ("accepted", None),
        "https://spam.test/ads": ("rejected", "off_topic"),
    }
    bundle = json.loads(vfs.read("run", f"research/{todo.todo_id}/evidence/result.json"))
    assert [s["locator"] for s in bundle["sources"]] == ["https://docs.test/locking"]


def test_page_accepted_by_two_parallel_tasks_is_stored_once(stores):
    vfs, index = stores
    todos = [ResearchTodo(research_question=f"Shared question {n}") for n in range(2)]
    shared = {"url": "https://shared.test/page", "content": "shared facts", "raw_content": "shared facts " * 80}

    supervisor = ResearchSupervisor(
        _researcher(FakeResearchModel(), lambda: ControlledMCP([shared])),
        vfs,
        index,
        max_concurrent_researchers=2,
    )
    results = supervisor.delegate("shared", todos)

    captures = [
        json.loads(vfs.read("shared", f"research/{todo.todo_id}/evidence/result.json"))["sources"]
        for todo, _, _ in results
    ]
    assert all(len(sources) == 1 for sources in captures)
    stored = [s for sources in captures for s in sources if not s.get("reused_from_task")]
    assert len(stored) == 1  # one capture; the other task references it
    locators = {c.metadata["locator"] for c in index.documents("shared")}
    assert locators == {"https://shared.test/page"}
    chunk_ids = [c.chunk_id for c in index.documents("shared")]
    assert len(chunk_ids) == len(set(chunk_ids))
