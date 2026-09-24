import hashlib
from intelligence_collector.contracts import VerificationResult
from intelligence_collector.publication import approval_for, can_request_approval, publish

def test_only_verified_artifact_can_request_approval():
    assert not can_request_approval(VerificationResult(artifact_hash="", passed=False))
    assert can_request_approval(VerificationResult(artifact_hash="hash", passed=True))

def test_rejection_and_stale_hash_leave_output_unchanged(tmp_path):
    content = "report"; digest = hashlib.sha256(content.encode()).hexdigest(); target = tmp_path / "out.md"
    verification = VerificationResult(artifact_hash=digest, passed=True)
    rejected = approval_for(digest, "v", False, "user")
    assert not publish(content, str(target), rejected, verification).published and not target.exists()
    stale = approval_for("stale", "v", True, "user")
    assert not publish(content, str(target), stale, verification).published and not target.exists()
