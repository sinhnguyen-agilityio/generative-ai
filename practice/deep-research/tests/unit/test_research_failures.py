"""Technical research failures: a limitation when other tasks have evidence, review otherwise."""

from intelligence_collector.contracts import ResearchTodo, TodoStatus
from intelligence_collector.graph.nodes import Node, ResearchNodes
from intelligence_collector.graph.routing import route_after_research
from intelligence_collector.storage.vfs import VFS


def _nodes(tmp_path, *, usable: bool):
    nodes = object.__new__(ResearchNodes)
    nodes.vfs = VFS(tmp_path)
    nodes.research_assessor = None
    failed = ResearchTodo(
        todo_id="broken", research_question="What changed in v2?", status=TodoStatus.FAILED
    )
    nodes.vfs.write_json(
        "r",
        "research/broken/outcome.json",
        {
            "status": "failed",
            "question": failed.research_question,
            "error": "Research sub-agent failed: 401 Unauthorized",
            "failure_kind": "technical",
        },
        "research_outcome",
        task_id="broken",
    )
    todos, evidence = [failed], []
    if usable:
        good = ResearchTodo(
            todo_id="good", research_question="What is the library?", status=TodoStatus.COMPLETED
        )
        ref = nodes.vfs.write_json(
            "r",
            "research/good/evidence/result.json",
            {
                "task_id": "good",
                "question": good.research_question,
                "summary": "An overview.",
                "sources": [{"url": "https://docs.test/a", "excerpt": "Overview"}],
                "sufficient": True,
                "unresolved_gaps": [],
            },
            "evidence_bundle",
            task_id="good",
        )
        todos.append(good)
        evidence.append(ref.model_dump(mode="json"))
    return nodes, {
        "run_id": "r",
        "todos": [todo.model_dump(mode="json") for todo in todos],
        "evidence": evidence,
    }


def test_one_technical_failure_does_not_stop_a_run_with_usable_evidence(tmp_path):
    nodes, state = _nodes(tmp_path, usable=True)
    result = nodes.research_complete(state)
    assert result["research_status"] == "PARTIAL"
    assert route_after_research(result) == Node.SYNTHESIZE
    assert "Research failed technically for: What changed in v2?" in result["research_limitations"]


def test_technical_failure_without_usable_evidence_goes_straight_to_review(tmp_path):
    nodes, state = _nodes(tmp_path, usable=False)
    result = nodes.research_complete(state)
    assert result["verification_issue"] == "research_failure"
    # No retry of the assessment: re-reading the same stored failure cannot change it.
    assert route_after_research(result) == Node.NEEDS_REVIEW
    # The recovery "research" action reopens exactly the failed task.
    assert [gap["todo_id"] for gap in result["research_gaps"]] == ["broken"]


def test_no_todos_goes_straight_to_review(tmp_path):
    nodes, _ = _nodes(tmp_path, usable=False)
    result = nodes.research_complete({"run_id": "r", "todos": []})
    assert route_after_research(result) == Node.NEEDS_REVIEW
