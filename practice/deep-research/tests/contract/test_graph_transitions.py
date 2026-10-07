import json

import pytest
from langchain_core.messages import AIMessage

from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.contracts import (
    ResearchTodo,
    Stage,
    TodoStatus,
    VerificationResult,
)
from intelligence_collector.graph.build import build_graph, node_wrapper
from intelligence_collector.graph.routing import PASSED, route_after_verify
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.publication import can_request_approval
from intelligence_collector.storage.vfs import VFS


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
    model = SequencedModel(
        [
            {"allowed": True, "reason": ""},
            {
                "need_clarification": True,
                "question": "Which databases and workload should be compared?",
                "verification": "",
            },
        ]
    )
    graph = build_graph(
        scope_agent=ScopeAgent(ModelRouter(model)),
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

    model = SequencedModel(
        [
            {"allowed": True, "reason": ""},
            _clear_scope(),
            _brief(),
            {
                "questions": [
                    "Which databases suit local Python applications, per primary sources?"
                ],
                "evidence_points": [[{"text": "evidence answering the question", "required_answers": [1]}]],
            },
        ]
    )
    graph = build_graph(
        scope_agent=ScopeAgent(ModelRouter(model)),
        vfs=PlanningWriteFailureVFS(tmp_path),
    )

    with pytest.raises(OSError, match="todo persistence failure"):
        graph.invoke(
            {"run_id": "todo-failure", "thread_id": "thread", "request": "Bounded request"}
        )

    assert not (tmp_path / "todo-failure" / "planning" / "todos.v1.json").exists()
    assert not (tmp_path / "todo-failure" / "research").exists()


def test_insufficient_research_gets_only_one_automatic_retry():
    from intelligence_collector.graph.routing import route_after_research
    from intelligence_collector.graph.nodes import Node

    assert route_after_research({"research_status": "INSUFFICIENT"}) == Node.PREPARE_GAP_RESEARCH
    assert (
        route_after_research({"research_status": "INSUFFICIENT", "research_gap_attempt": 1})
        == Node.SYNTHESIZE
    )


def test_unverified_result_cannot_traverse_to_approval_or_publication():
    result = VerificationResult(
        artifact_hash="candidate-hash",
        passed=False,
        repair_details="The draft omits a required comparison.",
    )

    route = route_after_verify({"verification": result.model_dump(mode="json")})
    assert route != PASSED  # a failed report never reaches faithfulness, writer or approval
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
        Stage.INCOMPLETE,
        Stage.CANCELLED,
        Stage.EXPORTED_INCOMPLETE,
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


def test_node_lifecycle_reports_waiting_on_interrupt():
    from langgraph.errors import GraphInterrupt

    events = []

    def pause(_state):
        raise GraphInterrupt(())

    wrapped = node_wrapper(
        "await_plan_approval",
        pause,
        lifecycle=lambda node, status: events.append((node, status)),
    )

    with pytest.raises(GraphInterrupt):
        wrapped({"run_id": "run", "thread_id": "thread"})

    assert events == [("await_plan_approval", "running"), ("await_plan_approval", "waiting")]


def test_every_research_round_ends_with_gap_analysis():
    from intelligence_collector.graph.routing import route_after_delegation, route_after_gap_analysis
    from intelligence_collector.graph.nodes import Node

    assert route_after_delegation({}) == Node.ANALYZE_GAPS
    assert route_after_delegation({"research_replan": {"reason": "x"}}) == Node.REPLAN
    assert route_after_gap_analysis({"gap_analysis": {"ended": "follow_ups_created"}}) == Node.RESEARCH
    for ended in ("no_essential_gaps", "caps_reached", "stalled", "turn_limit"):
        assert route_after_gap_analysis({"gap_analysis": {"ended": ended}}) == Node.ASSESS_RESEARCH
    assert route_after_gap_analysis({"research_replan": {"reason": "x"},
                                     "gap_analysis": {"ended": "follow_ups_created"}}) == Node.REPLAN
