from __future__ import annotations
import hashlib, os, tempfile
from pathlib import Path
from intelligence_collector.contracts import ApprovalDecision, PublicationRecord, VerificationResult, now
def publish(content: str, destination: str, approval: ApprovalDecision, verification: VerificationResult):
    digest = hashlib.sha256(content.encode()).hexdigest()
    if not verification.passed or approval.artifact_hash != digest or not approval.approved:
        return PublicationRecord(destination=destination, artifact_hash=digest, approval_id=approval.approval_id, published=False, error="approval/hash/verification mismatch")
    target=Path(destination); target.parent.mkdir(parents=True, exist_ok=True); fd,tmp=tempfile.mkstemp(dir=target.parent)
    with os.fdopen(fd,"w",encoding="utf-8") as f: f.write(content); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,target)
    return PublicationRecord(destination=destination, artifact_hash=digest, approval_id=approval.approval_id, published=True, published_at=now())

