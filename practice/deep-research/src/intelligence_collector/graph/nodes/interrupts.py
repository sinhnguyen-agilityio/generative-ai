"""Human-in-the-loop pauses. Each node calls LangGraph's ``interrupt()`` and does nothing
else before it, so a resume (``Command(resume=...)``) never repeats work. The payload's
``kind`` names the decision the graph waits for (see ``graph/recovery.py``)."""

from __future__ import annotations

from intelligence_collector.contracts import (
    ClarificationResumeDecision,
    PlanApprovalDecision,
    RecoveryAction,
    RecoveryDecision,
    ScopeAssessment,
    Stage,
)
from intelligence_collector.graph.state import GraphState


def await_clarification(state: GraphState) -> GraphState:
    """Pause after scope analysis and feed the validated answer back into scoping.

    Keeping this interrupt in its own node prevents the Scope Agent call in
    ``clarify_with_user`` from being repeated when LangGraph re-enters the interrupted node.
    """
    from langgraph.types import interrupt

    assessment = ScopeAssessment.model_validate(state["scope_assessment"])
    decision = ClarificationResumeDecision.model_validate(
        interrupt(
            {
                "kind": "clarification_required",
                "run_id": state.get("run_id"),
                "question": assessment.question,
            }
        )
    )
    return {
        **state,
        "request": decision.reply,
        "reviewer": decision.actor,
        "current_action": "Clarification received — reassessing scope",
    }


def await_plan_approval(state: GraphState) -> GraphState:
    """Pause before any retrieval until a human approves the exact plan (initial or replacement).

    No model call, tool call or artifact write happens before ``interrupt()``; the approval is
    bound to the plan hash, so a changed plan always needs a new approval.
    """
    from langgraph.types import interrupt

    plan = state.get("plan") or {}
    decision = PlanApprovalDecision.model_validate(
        interrupt(
            {
                "kind": "plan_approval",
                "run_id": state.get("run_id"),
                "plan": plan,
            }
        )
    )
    outcome = decision.outcome
    record = {
        "plan_hash": plan.get("plan_hash", ""),
        "approved": outcome == "approve",
        "outcome": outcome,
        "actor": decision.actor,
        "kind": plan.get("kind", "initial"),
    }
    update = {**state, "plan_approval": record, "reviewer": decision.actor}
    if outcome == "approve":
        return {
            **update,
            "stage": Stage.RESEARCHING.value,
            "status": "active",
            "current_action": "Plan approved — starting research",
        }
    if outcome == "change":
        # The requested change becomes the next scope turn: new brief, new plan, new approval.
        return {
            **update,
            "stage": Stage.SCOPING.value,
            "request": decision.reply,
            "current_action": "Plan change requested — rescoping",
        }
    if plan.get("kind") == "replacement":
        return {
            **update,
            "stage": Stage.NEEDS_REVIEW.value,
            "current_action": "Replacement plan declined",
            "last_error": "The replacement research plan was declined.",
        }
    return {
        **update,
        "stage": "Cancelled",
        "status": "cancelled",
        "current_action": "Plan declined; run cancelled",
    }


def await_recovery(state: GraphState, *, can_rescope=True) -> GraphState:
    """Pause a failed BVF run for an explicit, validated human recovery decision.

    The next node applies the stored decision through an explicit conditional graph edge.
    """
    from langgraph.types import interrupt

    decision = RecoveryDecision.model_validate(
        interrupt(
            {
                "kind": "recovery_required",
                "run_id": state.get("run_id"),
                "reason": state.get("current_action") or state.get("last_error", ""),
                "allowed_actions": [
                    action.value
                    for action in RecoveryAction
                    if can_rescope or action is not RecoveryAction.RESCOPE
                ],
            }
        )
    )
    return {
        **state,
        "recovery_decision": decision.model_dump(mode="json"),
        "current_action": f"Recovery action authorized: {decision.action.value}",
    }


def approval(state: GraphState) -> GraphState:
    """Pause the graph until a human resumes it with a publication decision."""
    from langgraph.types import interrupt

    decision = interrupt({"kind": "publication_approval", "run_id": state.get("run_id")})
    stage = Stage.PUBLISHED if decision.get("approved") else Stage.AWAITING_APPROVAL
    return {**state, "approval": decision, "stage": stage.value}
