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
    return ResearchAgent(ModelRouter(model, mode="fake", **kwargs), mcp_factory)


def test_research_agent_searches_broad_then_narrow_and_persists_provenance(stores):
    vfs, index = stores
    todo = ResearchTodo(research_question="Compare database transaction behavior")
    narrow = "database locking behavior primary documentation"
    model = FakeResearchModel(queries={todo.research_question: [todo.research_question, narrow]})
    mcp = ControlledMCP([
        [{"url": "https://primary.test/overview", "content": "Transaction overview"}],
        [{"url": "https://primary.test/locking", "content": "Documented locking behavior"}],
    ])
    traces = []

    completed, refs, chunks = run_worker(
        vfs, index, "run", todo,
        _researcher(model, lambda: mcp, trace=lambda **event: traces.append(event)),
    )

    assert completed.status == TodoStatus.COMPLETED
    assert [request["query"] for request in mcp.requests] == [todo.research_question, narrow]
    assert model.bound_tools[0]["function"]["name"] == "tavily_search"
    assert [event["role"] for event in traces] == [
        "researcher", "researcher", "researcher", "research_compressor",
    ]
    assert traces[0]["prompt_version"] == "researcher.v1"
    assert traces[-1]["prompt_version"] == "research-compress.v1"
    assert {ref.kind for ref in refs} == {"raw_capture", "notes", "excerpts", "evidence_bundle"}
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

    completed, _, _ = run_worker(vfs, index, "early", todo, _researcher(FakeResearchModel(), lambda: mcp))

    assert completed.status == TodoStatus.COMPLETED
    assert mcp.calls == 1


def test_search_budget_is_a_hard_per_task_limit_and_gaps_are_recorded(stores):
    vfs, index = stores
    todo = ResearchTodo(research_question="Investigate an unresolved compatibility issue")
    model = FakeResearchModel(
        queries={todo.research_question: [f"query {n}" for n in range(8)]},
        findings={todo.research_question: {
            "summary": "Partial evidence [https://primary.test/partial]",
            "sufficient": False,
            "unresolved_gaps": ["cross-platform confirmation"],
        }},
    )
    mcp = ControlledMCP([{"url": "https://primary.test/partial", "content": "Partial evidence"}])

    completed, _, _ = run_worker(vfs, index, "limited", todo, _researcher(model, lambda: mcp))

    evidence = json.loads(vfs.read("limited", f"research/{todo.todo_id}/evidence/result.json"))
    assert mcp.calls == 5
    assert completed.status == TodoStatus.COMPLETED  # best available outcome, gap recorded
    assert evidence["sufficient"] is False
    assert evidence["unresolved_gaps"] == ["cross-platform confirmation"]
    assert evidence["search_calls"] == 5


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
        queries={todo.research_question: [f"{todo.research_question} q{i}" for i in range(4)]
                 for todo in todos},
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


def test_failing_sub_agent_does_not_cancel_siblings(stores):
    vfs, index = stores
    ok, broken = ResearchTodo(research_question="Works"), ResearchTodo(research_question="Breaks")
    model = FakeResearchModel(fail_questions={"Breaks"})
    supervisor = ResearchSupervisor(
        _researcher(model, lambda: ControlledMCP([{"url": "https://ok.test", "content": "x"}])),
        vfs,
        index,
    )

    results = {todo.research_question: todo.status for todo, _, _ in supervisor.delegate("mixed", [ok, broken])}

    assert results == {"Works": TodoStatus.COMPLETED, "Breaks": TodoStatus.FAILED}
    outcome = json.loads(vfs.read("mixed", f"research/{broken.todo_id}/outcome.json"))
    assert "Research sub-agent failed" in outcome["error"]
    assessment = supervisor.assess([ok, broken], ["artifact"], ["chunk"])
    assert assessment.ready_for_synthesis is False
    assert assessment.blockers == ["Task not completed (failed): Breaks"]
