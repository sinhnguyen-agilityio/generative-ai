"""Conditional edges of the graph: each router reads the state and names the next node (see
``build.py``).
"""

from intelligence_collector.graph.nodes import (
    DEFAULT_RECOVERY_BUDGET,
    MAX_STALLED_ATTEMPTS,
    Node,
)
from intelligence_collector.graph.state import GraphState
from intelligence_collector.contracts import RecoveryAction, RepairRoute, VerificationIssue

# Outcomes of route_after_verify; build.py maps "passed" to approval or END.
PASSED, FIX, NEEDS_REVIEW = "passed", "fix", "needs_review"

# Outcome of route_after_publish when the report was written; build.py maps it to END.
PUBLISHED = "published"

# Outcomes of route_after_approval; build.py maps these to PUBLISH or CLASSIFY_FEEDBACK.
APPROVED, REJECTED = "approved", "rejected"

# Outcomes of route_after_feedback; build.py maps these to PREPARE_REVISION, LOCATE_EXTENSION
# or CLARIFY.
TO_REVISE, TO_EXTEND, TO_SCOPE = "revise", "extend", "scope"

# Outcomes of route_after_plan_approval.
PLAN_APPROVED, PLAN_CHANGE, PLAN_DECLINED, REPLACEMENT_DECLINED = (
    "approve",
    "change",
    "decline",
    "replacement_declined",
)

# ===== CONDITIONAL EDGE ROUTERS =====


def route_after_recovery(state: GraphState) -> str:
    """Only recognized human actions can select a recovery transition."""
    return RecoveryAction(state.get("recovery_route")).value


def route_after_clarify(state: GraphState) -> str:
    """Pause at the dedicated clarification interrupt, or continue to the brief."""
    assessment = state.get("scope_assessment") or {}
    if assessment.get("need_clarification", True):
        return Node.AWAIT_CLARIFICATION
    return Node.WRITE_BRIEF


def route_after_plan_approval(state: GraphState) -> str:
    """Approved plans start retrieval; a change request rescopes; a declined initial plan ends
    the run and a declined replacement plan goes to review with the latest draft."""
    approval = state.get("plan_approval") or {}
    outcome = approval.get("outcome")
    if outcome == "approve" and approval.get("approved"):
        return PLAN_APPROVED
    if outcome == "change":
        return PLAN_CHANGE
    return REPLACEMENT_DECLINED if approval.get("kind") == "replacement" else PLAN_DECLINED


def route_after_verify(state: GraphState) -> str:
    """PASS leaves the loop at once. ERROR is retried, then escalated; it never rewrites the
    report. FAIL opens an improvement cycle while the budget lasts and progress is made."""
    verification = state.get("verification") or {}
    if verification.get("passed"):
        return PASSED
    issue = state.get("verification_issue", VerificationIssue.DRAFT_DEFECT)
    if issue == VerificationIssue.VERIFIER_FAILURE:
        return Node.RETRY_VERIFIER if state.get("verifier_attempt", 0) < 2 else NEEDS_REVIEW
    budget = state.get("recovery_budget", DEFAULT_RECOVERY_BUDGET)
    if state.get("recovery_attempt", 0) >= budget:
        return NEEDS_REVIEW
    if state.get("stall_count", 0) >= MAX_STALLED_ATTEMPTS:
        return NEEDS_REVIEW
    return FIX


def route_after_fix(state: GraphState) -> str:
    """Follow the diagnosed repair route."""
    route = (state.get("repair") or {}).get("route")
    if route == RepairRoute.RESEARCH:
        return Node.PREPARE_REPAIR_RESEARCH
    if route == RepairRoute.REPLAN:
        return Node.REPLAN
    return Node.SYNTHESIZE


def route_after_delegation(state: GraphState) -> str:
    """A supervisor's replan proposal goes to replacement planning; otherwise assess research."""
    return Node.REPLAN if state.get("research_replan") else Node.ASSESS_RESEARCH


def route_after_research(state: GraphState) -> str:
    """A technical failure with no usable evidence goes to review; ready evidence (or a spent
    gap budget) goes to synthesis; otherwise research the gaps once."""
    if state.get("verification_issue") == VerificationIssue.RESEARCH_FAILURE:
        return Node.NEEDS_REVIEW
    if (state.get("progress_assessment") or {}).get("ready_for_synthesis"):
        return Node.SYNTHESIZE
    if state.get("research_gap_attempt", 0) >= min(state.get("research_gap_budget", 1), 1):
        return Node.SYNTHESIZE
    return Node.PREPARE_GAP_RESEARCH


def route_after_publish(state: GraphState) -> str:
    """A written report ends the run; a refused or failed write goes to review."""
    return PUBLISHED if (state.get("publication") or {}).get("published") else Node.NEEDS_REVIEW


def route_after_approval(state: GraphState) -> str:
    """Post-HITL-review routing: an explicit approval publishes; a chat reply or a rejection
    goes to the reply classifier."""
    return APPROVED if (state.get("approval") or {}).get("approved") else REJECTED


def route_after_feedback(state: GraphState) -> str:
    """Send the classified reply to a rewrite (revise), an added section (extend) or back to
    scoping for new research (scope). Anything unrecognised is a revision."""
    route = state.get("feedback_route")
    if route in {TO_EXTEND, TO_SCOPE}:
        return route
    return TO_REVISE


def route_after_extension(state: GraphState) -> str:
    """Stored evidence covers the requested section -> write it; otherwise research the one
    added task first."""
    return Node.SYNTHESIZE if (state.get("extension") or {}).get("sufficient") else Node.RESEARCH
