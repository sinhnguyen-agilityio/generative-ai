from langgraph.graph import END

from intelligence_collector.agents.recovery import choose_repair
from intelligence_collector.contracts import IssueType, Stage
from intelligence_collector.graph.nodes import (
    DEFAULT_RECOVERY_BUDGET,
    MAX_STALLED_ATTEMPTS,
    Node,
)
from intelligence_collector.graph.state import GraphState

# Outcomes of route_after_verify; build.py maps "passed" to approval or END.
PASSED, FIX, NEEDS_REVIEW = "passed", "fix", "needs_review"

# Outcomes of route_after_approval; build.py maps these to PUBLISH or CLASSIFY_FEEDBACK.
APPROVED, REJECTED = "approved", "rejected"

# Outcomes of route_after_feedback; build.py maps these to PUBLISH, WRITER or CLARIFY.
TO_PUBLISH, TO_FORMAT, TO_SCOPE = "approve", "format", "scope"

# ===== CONDITIONAL EDGE ROUTERS =====


def route_after_clarify(state: GraphState) -> str:
    """End the turn to wait for the user's answer, or continue to the research brief."""
    assessment = state.get("scope_assessment") or {}
    if assessment.get("need_clarification", True):
        return END
    return Node.WRITE_BRIEF


def route_after_verify(state: GraphState) -> str:
    """BVF gate: only a fully passing report leaves the loop; otherwise fix or escalate."""
    verification = state.get("verification") or {}
    if verification.get("passed"):
        return PASSED
    budget = state.get("recovery_budget", DEFAULT_RECOVERY_BUDGET)
    if state.get("recovery_attempt", 0) >= budget:
        return NEEDS_REVIEW
    if state.get("stall_count", 0) >= MAX_STALLED_ATTEMPTS:
        return NEEDS_REVIEW
    return FIX


def route_after_approval(state: GraphState) -> str:
    """Post-HITL-review routing: an explicit approval publishes; a chat reply or a rejection
    goes to the reply classifier."""
    return APPROVED if (state.get("approval") or {}).get("approved") else REJECTED


def route_after_feedback(state: GraphState) -> str:
    """Send the classified reply to publishing (approve), the writer (format) or back to
    scoping (scope)."""
    route = state.get("feedback_route")
    if route == "approve" and (state.get("approval") or {}).get("approved"):
        return TO_PUBLISH
    return TO_SCOPE if route == "scope" else TO_FORMAT


def route_after_fix(state: GraphState) -> str:
    """Send each failure type to its repair: redraft, or research the gaps first."""
    issue = (state.get("verification") or {}).get("issue_type")
    if issue == IssueType.DRAFT_DEFECT.value:
        return Node.SYNTHESIZE
    if issue == IssueType.EVIDENCE_GAP.value:
        return Node.RESEARCH_GAPS
    return Node.NEEDS_REVIEW  # PLAN_GAP is never emitted by the source-verification script


# ===== RECOVERY ROUTING =====


def route_verification(result):
    if result.passed:
        return Stage.AWAITING_APPROVAL
    return {
        IssueType.DRAFT_DEFECT: Stage.SYNTHESIZING,
        IssueType.EVIDENCE_GAP: Stage.RESEARCHING,
        IssueType.PLAN_GAP: Stage.PLANNING}[result.issue_type]


def route_repair(state, result, fingerprint):
    return choose_repair(state, result.issue_type, fingerprint)
