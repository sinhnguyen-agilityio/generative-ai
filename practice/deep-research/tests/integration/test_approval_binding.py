import hashlib
from intelligence_collector.contracts import ApprovalDecision, VerificationResult
from intelligence_collector.publication import publish
def test_publication_requires_exact_approved_hash(tmp_path):
    content = "verified report"; digest = hashlib.sha256(content.encode()).hexdigest()
    verification = VerificationResult(artifact_hash=digest, passed=True)
    bad = ApprovalDecision(artifact_hash="stale", verification_id=verification.verification_id, approved=True, actor="user")
    target = tmp_path / "report.md"
    assert not publish(content, str(target), bad, verification).published and not target.exists()
    good = ApprovalDecision(artifact_hash=digest, verification_id=verification.verification_id, approved=True, actor="user")
    assert publish(content, str(target), good, verification).published and target.read_text() == content


def test_reformatted_report_publishes_only_when_bound_to_its_verified_source(tmp_path):
    source = "verified report"
    formatted = "- verified report"
    verification = VerificationResult(artifact_hash=hashlib.sha256(source.encode()).hexdigest(), passed=True)
    approval = ApprovalDecision(
        artifact_hash=hashlib.sha256(formatted.encode()).hexdigest(),
        verification_id=verification.verification_id, approved=True, actor="user",
    )
    target = tmp_path / "report.md"
    # Without its verified source, a reformatted text is just an unverified artifact.
    assert not publish(formatted, str(target), approval, verification).published and not target.exists()
    assert not publish(formatted, str(target), approval, verification, verified_source="tampered").published
    assert publish(formatted, str(target), approval, verification, verified_source=source).published
    assert target.read_text() == formatted
