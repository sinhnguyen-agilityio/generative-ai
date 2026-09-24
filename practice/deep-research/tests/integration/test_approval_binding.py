import hashlib
from intelligence_collector.contracts import ApprovalDecision, VerificationResult
from intelligence_collector.publication import publish
def test_publication_requires_exact_approved_hash(tmp_path):
    content = "verified report"; digest = hashlib.sha256(content.encode()).hexdigest()
    verification = VerificationResult(artifact_hash=digest, passed=True)
    bad = ApprovalDecision(artifact_hash="stale", verification_id="v", approved=True, actor="user")
    target = tmp_path / "report.md"
    assert not publish(content, str(target), bad, verification).published and not target.exists()
    good = ApprovalDecision(artifact_hash=digest, verification_id="v", approved=True, actor="user")
    assert publish(content, str(target), good, verification).published and target.read_text() == content
