from intelligence_collector.contracts import IssueType, Stage
def route_verification(result):
    if result.passed: return Stage.AWAITING_APPROVAL
    return {IssueType.DRAFT_DEFECT: Stage.SYNTHESIZING, IssueType.EVIDENCE_GAP: Stage.RESEARCHING, IssueType.PLAN_GAP: Stage.PLANNING}[result.issue_type]

