import json

import pytest
from langchain_core.messages import AIMessage

from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.contracts import (
    IssueType,
    ResearchTodo,
    Stage,
    TodoStatus,
    VerificationResult,
)
from intelligence_collector.graph.build import build_graph, node_wrapper
from intelligence_collector.graph.routing import route_verification
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.publication import can_request_approval
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.tools.think_tool import think_tool


class SequencedModel:
    model_name = "graph-transition-fake"

    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, messages, config=None):
        return AIMessage(content=json.dumps(next(self.responses)))


def _clear_scope():
    return {
        "need_clarification": False,
        "question": "",
        "verification": "The scope is clear; research will now begin.",
    }


def _brief():
    return {
        "research_question": (
            "I want to compare databases for local Python applications using primary technical "
            "sources, with every factual claim cited, and receive a technical report."
        )
    }


def test_invalid_scope_cannot_traverse_to_planning(tmp_path):
    model = SequencedModel([
        {
            "need_clarification": True,
            "question": "Which databases and workload should be compared?",
            "verification": "",
        }
    ])
    graph = build_graph(
        scope_agent=ScopeAgent(ModelRouter(model, mode="live")),
        vfs=VFS(tmp_path),
    )

    result = graph.invoke(
        {"run_id": "invalid-scope", "thread_id": "thread", "request": "Compare databases"}
    )

    assert result["stage"] == "Scoping"
    assert result["scope_assessment"]["need_clarification"] is True
    assert "scoped_request" not in result
    assert "todos" not in result


def test_failed_todo_write_cannot_traverse_to_research(tmp_path):
    class PlanningWriteFailureVFS(VFS):
        def write_json(self, run_id, relative_path, payload, kind, **kwargs):
            if relative_path == "planning/todos.v1.json":
                raise OSError("simulated todo persistence failure")
            return super().write_json(run_id, relative_path, payload, kind, **kwargs)

    model = SequencedModel([_clear_scope(), _brief()])
    graph = build_graph(
        scope_agent=ScopeAgent(ModelRouter(model, mode="live")),
        vfs=PlanningWriteFailureVFS(tmp_path),
    )

    with pytest.raises(OSError, match="todo persistence failure"):
        graph.invoke(
            {"run_id": "todo-failure", "thread_id": "thread", "request": "Bounded request"}
        )

    assert not (tmp_path / "todo-failure" / "planning" / "todos.v1.json").exists()
    assert not (tmp_path / "todo-failure" / "research").exists()


def test_incomplete_research_cannot_traverse_to_synthesis():
    todos = [ResearchTodo(research_question="Investigate database behavior")]

    assessment = think_tool(todos, artifact_count=0, index_count=0)

    assert assessment.ready_for_synthesis is False
    assert assessment.next_action == "continue research"
    assert todos[0].status == TodoStatus.PLANNED


def test_unverified_result_cannot_traverse_to_approval_or_publication():
    result = VerificationResult(
        artifact_hash="candidate-hash",
        passed=False,
        issue_type=IssueType.DRAFT_DEFECT,
        repair_details="The draft omits a required comparison.",
    )

    assert route_verification(result) == Stage.SYNTHESIZING
    assert route_verification(result) != Stage.AWAITING_APPROVAL
    assert can_request_approval(result) is False


def test_workflow_stage_contract_distinguishes_all_guarded_states():
    assert {
        Stage.SCOPING,
        Stage.PLANNING,
        Stage.RESEARCHING,
        Stage.SYNTHESIZING,
        Stage.VERIFYING,
        Stage.AWAITING_APPROVAL,
        Stage.PUBLISHED,
        Stage.FAILED,
        Stage.NEEDS_REVIEW,
    } == set(Stage)


def test_node_lifecycle_reports_running_and_terminal_status():
    events = []
    wrapped = node_wrapper(
        "planning",
        lambda state: {**state, "stage": "Planning"},
        lifecycle=lambda node, status: events.append((node, status)),
    )

    wrapped({"run_id": "run", "thread_id": "thread"})

    assert events == [("planning", "running"), ("planning", "success")]


def test_node_lifecycle_reports_failure():
    events = []

    def fail(_state):
        raise RuntimeError("planned failure")

    wrapped = node_wrapper(
        "planning",
        fail,
        lifecycle=lambda node, status: events.append((node, status)),
    )

    with pytest.raises(RuntimeError, match="planned failure"):
        wrapped({"run_id": "run", "thread_id": "thread"})

    assert events == [("planning", "running"), ("planning", "failure")]
