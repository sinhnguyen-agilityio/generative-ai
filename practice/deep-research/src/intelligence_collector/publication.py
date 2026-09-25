from __future__ import annotations
import hashlib
import os
import tempfile
from pathlib import Path
from intelligence_collector.contracts import ApprovalDecision, PublicationRecord, VerificationResult, now


def can_request_approval(verification: VerificationResult) -> bool:
    return verification.passed and bool(verification.artifact_hash)


def approval_for(artifact_hash: str, verification_id: str, approved: bool, actor: str) -> ApprovalDecision:
    return ApprovalDecision(artifact_hash=artifact_hash, verification_id=verification_id, approved=approved, actor=actor)


def publish(content: str, destination: str, approval: ApprovalDecision, verification: VerificationResult):
    digest = hashlib.sha256(content.encode()).hexdigest()
    authorized = (
        verification.passed
        and approval.approved
        and approval.artifact_hash == digest == verification.artifact_hash
        and approval.verification_id == verification.verification_id
    )
    if not authorized:
        return PublicationRecord(destination=destination, artifact_hash=digest, approval_id=approval.approval_id, published=False, error="approval/hash/verification mismatch")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return PublicationRecord(destination=destination, artifact_hash=digest, approval_id=approval.approval_id, published=True, published_at=now())
