"""Conditional edges of the graph: each router reads the state and names the next node (see
``build.py``).
"""

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

# Outcomes of route_after_feedback; build.py maps these to PUBLISH, SYNTHESIZE or CLARIFY.
TO_PUBLISH, TO_REVISE, TO_SCOPE = "approve", "revise", "scope"

# ===== CONDITIONAL EDGE ROUTERS =====


def route_after_clarify(state: GraphState) -> str:
    """Pause at the dedicated clarification interrupt, or continue to the brief."""
    assessment = state.get("scope_assessment") or {}
    if assessment.get("need_clarification", True):
        return Node.AWAIT_CLARIFICATION
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
    """Send the classified reply to publishing (approve), a rewrite (revise) or back to scoping
    for new research (scope). Anything unrecognised is a revision, never a publication."""
    route = state.get("feedback_route")
    if route == "approve" and (state.get("approval") or {}).get("approved"):
        return TO_PUBLISH
    return TO_SCOPE if route == "scope" else TO_REVISE
