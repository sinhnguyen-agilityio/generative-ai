from intelligence_collector.contracts import IssueType, Stage
from intelligence_collector.agents.recovery import choose_repair


def route_verification(result):
    if result.passed:
        return Stage.AWAITING_APPROVAL
    return {
        IssueType.DRAFT_DEFECT: Stage.SYNTHESIZING,
        IssueType.EVIDENCE_GAP: Stage.RESEARCHING,
        IssueType.PLAN_GAP: Stage.PLANNING}[result.issue_type]


def route_repair(state, result, fingerprint):
    return choose_repair(state, result.issue_type, fingerprint)
