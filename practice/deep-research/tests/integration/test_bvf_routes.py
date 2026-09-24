from intelligence_collector.contracts import IssueType, VerificationResult, Stage
from intelligence_collector.graph.routing import route_verification
def test_bvf_routes():
    for issue, stage in ((IssueType.DRAFT_DEFECT, Stage.SYNTHESIZING), (IssueType.EVIDENCE_GAP, Stage.RESEARCHING), (IssueType.PLAN_GAP, Stage.PLANNING)):
        assert route_verification(VerificationResult(artifact_hash="x", passed=False, issue_type=issue)) == stage
