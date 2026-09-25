from langgraph.graph import END

from intelligence_collector.agents.recovery import choose_repair
from intelligence_collector.contracts import IssueType, Stage
from intelligence_collector.graph.nodes import Node
from intelligence_collector.graph.state import GraphState

# ===== CONDITIONAL EDGE ROUTERS =====


def route_after_clarify(state: GraphState) -> str:
    """End the turn to wait for the user's answer, or continue to the research brief."""
    assessment = state.get("scope_assessment") or {}
    if assessment.get("need_clarification", True):
        return END
    return Node.WRITE_BRIEF


def route_after_verify(state: GraphState) -> str:
    """Only a passed verification may request publication approval."""
    verification = state.get("verification") or {}
    return Node.APPROVE if verification.get("passed") else END


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
