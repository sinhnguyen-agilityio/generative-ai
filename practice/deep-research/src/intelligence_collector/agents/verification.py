from intelligence_collector.contracts import IssueType, VerificationResult
def verify(draft, scope, chunks):
    if not chunks: return VerificationResult(artifact_hash="", passed=False, issue_type=IssueType.EVIDENCE_GAP, repair_details="No evidence")
    required = (scope.objective,)
    if any(x not in draft for x in required): return VerificationResult(artifact_hash="", passed=False, issue_type=IssueType.DRAFT_DEFECT, repair_details="Objective missing")
    return VerificationResult(artifact_hash="", passed=True, checks=["scope", "evidence"])

def repair_draft(draft: str, feedback: str) -> str:
    return draft + f"\n\n## Verification repair\n{feedback}\n"

def preserve_evidence(evidence):
    return list(evidence)
