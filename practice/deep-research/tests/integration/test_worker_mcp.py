import json

import pytest

from intelligence_collector.agents.worker import ControlledMCP, run_worker
from intelligence_collector.contracts import ResearchTodo, TodoStatus


def test_worker_searches_broad_then_narrow_and_persists_provenance(stores):
    vfs, index = stores
    todo = ResearchTodo(research_question="Compare database transaction behavior")
    mcp = ControlledMCP([
        [{
            "url": "https://primary.test/overview",
            "content": "Transaction overview",
            "sufficient": False,
            "remaining_gaps": ["locking behavior"],
            "next_query": "database locking behavior primary documentation",
        }],
        [{
            "url": "https://primary.test/locking",
            "content": "Documented locking behavior",
            "sufficient": True,
        }],
    ])

    completed, refs, chunks = run_worker(vfs, index, "run", todo, mcp)

    assert completed.status == TodoStatus.COMPLETED
    assert mcp.calls == 2
    assert mcp.requests[0]["query"] == todo.research_question
    assert mcp.requests[1]["query"] == "database locking behavior primary documentation"
    assert {ref.kind for ref in refs} == {"raw_capture", "notes", "excerpts", "evidence_bundle"}
    evidence = json.loads(vfs.read("run", f"research/{todo.todo_id}/evidence/result.json"))
    assert evidence["sufficient"] is True
    assert evidence["sources"][0]["mcp_server"] == "controlled"
    assert evidence["sources"][0]["provenance"]["query"] == todo.research_question
    assert evidence["sources"][1]["provenance"]["query"] == (
        "database locking behavior primary documentation"
    )
    assert evidence["sources"][0]["content_hash"]
    assert all(chunk.metadata["locator"].startswith("https://") for chunk in chunks)


def test_worker_stops_early_when_first_search_is_sufficient(stores):
    vfs, index = stores
    todo = ResearchTodo(research_question="Find the transaction specification")
    mcp = ControlledMCP([{
        "url": "https://primary.test/spec",
        "content": "Complete specification",
        "sufficient": True,
    }])

    completed, _, _ = run_worker(vfs, index, "early", todo, mcp)

    assert completed.status == TodoStatus.COMPLETED
    assert mcp.calls == 1


def test_complex_worker_stops_at_five_calls_and_records_gap(stores):
    vfs, index = stores
    todo = ResearchTodo(research_question="Investigate an unresolved compatibility issue")
    response = [{
        "url": "https://primary.test/partial",
        "content": "Partial evidence",
        "sufficient": False,
        "remaining_gaps": ["cross-platform confirmation"],
    }]
    mcp = ControlledMCP([response] * 6, max_calls=5)

    completed, _, _ = run_worker(
        vfs, index, "limited", todo, mcp, complexity="complex"
    )

    evidence = json.loads(vfs.read(
        "limited", f"research/{todo.todo_id}/evidence/result.json"
    ))
    assert completed.status == TodoStatus.FAILED
    assert mcp.calls == 5
    assert evidence["sufficient"] is False
    assert evidence["unresolved_gaps"] == ["cross-platform confirmation"]


def test_mcp_failure_is_persisted_as_task_outcome(stores):
    class FailingMCP:
        max_calls = 5

        def search(self, query, **kwargs):
            raise TimeoutError("controlled timeout")

    vfs, index = stores
    todo = ResearchTodo(research_question="Find primary documentation")

    with pytest.raises(RuntimeError, match="MCP retrieval failed"):
        run_worker(vfs, index, "failed", todo, FailingMCP())

    outcome = json.loads(vfs.read(
        "failed", f"research/{todo.todo_id}/outcome.json"
    ))
    assert todo.status == TodoStatus.FAILED
    assert outcome["status"] == "failed"
    assert "TimeoutError" in outcome["error"]
