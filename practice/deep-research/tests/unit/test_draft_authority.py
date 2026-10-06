"""A new draft never inherits a verification, approval, publication or quality score earned by
an earlier draft: every node that writes a draft, or sends the run back to write one, merges
``invalidate_draft_authority()``."""

import pytest

from intelligence_collector.contracts import (
    FeedbackDecision,
    ResearchBrief,
    ResearchTodo,
    Stage,
    TodoStatus,
    WorkflowStatus,
)
from intelligence_collector.agents.worker import _evidence_path
from intelligence_collector.graph.instrumentation import node_wrapper
from intelligence_collector.graph.nodes import (
    ApprovalNodes,
    BVFNodes,
    PlannerNodes,
    RecoveryNodes,
    ResearchNodes,
    invalidate_draft_authority,
)
from intelligence_collector.graph.state import GraphState
from intelligence_collector.storage.vfs import VFS

BRIEF = ResearchBrief(research_question="What do HTTP 401 and 403 status codes mean?")

# State left behind by a draft that passed verification, was approved and scored.
STALE = {
    "verification": {"passed": True, "artifact_hash": "old", "verification_id": "v1"},
    "verification_outcome": "PASS",
    "coverage_cache": {"artifact_hash": "old", "checks": []},
    "verifier_attempt": 1,
    "verifier_error": "timeout",
    "approval": {"approved": True, "artifact_hash": "old"},
    "publication": {"published": True},
    "faithfulness_score": 0.97,
    "faithfulness_error": "old error",
    "context_precision_score": 0.8,
    "context_precision_error": "old error",
    "context_precision_details": [{"question": "q", "score": 0.8}],
    "response_relevancy_score": 0.9,
    "response_relevancy_error": "old error",
}


def _assert_no_authority(result: dict) -> None:
    for key, cleared in invalidate_draft_authority().items():
        assert key in result, f"{key} was not reset"
        assert result[key] == cleared, f"{key} kept {result[key]!r}"


def _state(**extra) -> dict:
    return {
        "run_id": "run",
        "thread_id": "thread",
        "draft": "Approved draft",
        "draft_version": 1,
        "scoped_request": BRIEF.model_dump(mode="json"),
        **STALE,
        **extra,
    }


def test_helper_clears_every_field_vouching_for_a_draft():
    update = invalidate_draft_authority()
    assert set(update) == set(STALE)
    assert set(update) <= set(GraphState.__annotations__)
    # A fresh dict each call, so callers can never share a mutable default.
    update["context_precision_details"].append("x")
    assert invalidate_draft_authority()["context_precision_details"] == []


def test_prepare_revision_drops_authority(tmp_path):
    nodes = object.__new__(BVFNodes)
    _assert_no_authority(nodes.prepare_revision(_state(reviewer_requests=["Use bullets."])))


@pytest.mark.parametrize("route", ["revise", "extend", "scope"])
def test_every_reviewer_feedback_route_drops_authority(tmp_path, route):
    class Router:
        def route(self, *args, **kwargs):
            return FeedbackDecision(route=route, topic="persistence")

    nodes = ApprovalNodes(VFS(tmp_path), str(tmp_path / "exports"), Router())
    _assert_no_authority(nodes.classify_feedback(_state(human_feedback="Add a section")))


@pytest.mark.parametrize(
    "decision",
    [
        {"action": "retry_report", "additional_attempts": 1},
        {"action": "revise", "feedback": "Use bullets"},
        {"action": "research", "feedback": "Find latency evidence"},
        {"action": "rescope", "feedback": "Compare only SQLite"},
    ],
)
def test_recovery_actions_that_lead_to_a_new_draft_drop_authority(tmp_path, decision):
    result = RecoveryNodes(VFS(tmp_path)).apply(_state(recovery_decision=decision))
    _assert_no_authority(result)


def test_gap_research_drops_authority(tmp_path):
    vfs = VFS(tmp_path)
    todo = ResearchTodo(todo_id="a", research_question="Question A", status=TodoStatus.COMPLETED)
    vfs.write_json("run", _evidence_path(todo), {"summary": "old"}, "evidence_bundle")
    nodes = object.__new__(ResearchNodes)
    nodes.vfs = vfs
    result = nodes.prepare_gap_research(
        _state(
            todos=[todo.model_dump(mode="json")],
            research_gaps=[{"todo_id": "a", "instruction": "Find the missing number"}],
        )
    )
    _assert_no_authority(result)


def test_new_plan_drops_the_draft_and_its_authority(tmp_path):
    class Planner:
        def plan(self, brief, vfs, run_id, thread_id=""):
            return [ResearchTodo(research_question="What does 401 mean?")], ""

    result = PlannerNodes(Planner(), VFS(tmp_path)).planning(_state())
    assert result["draft"] == ""
    _assert_no_authority(result)


def test_wrapper_merges_partial_updates_for_registry_and_persists_migrations():
    calls = []

    class Registry:
        def register_run(self, run_id, thread_id, **kwargs):
            pass

        def update_run(self, run_id, thread_id, **kwargs):
            calls.append((run_id, thread_id, kwargs["workflow_status"], kwargs["current_stage"]))

    wrapped = node_wrapper(
        "publishing", lambda state: {"stage": Stage.PUBLISHED.value}, run_registry=Registry()
    )
    # A legacy checkpoint: the migration renames the issue and fills defaults.
    result = wrapped(
        {"run_id": "run", "thread_id": "thread", "verification_issue": "research_verifier_failure"}
    )

    # The node's own keys win; migrated keys are written back; untouched keys are not repeated.
    assert result["stage"] == Stage.PUBLISHED.value
    assert result["verification_issue"] == "research_failure"
    assert result["research_gap_budget"] == 1
    assert "run_id" not in result
    # The registry sees the run identity from the state and the status after the update.
    assert calls == [("run", "thread", WorkflowStatus.PUBLISHED, Stage.PUBLISHED.value)]
