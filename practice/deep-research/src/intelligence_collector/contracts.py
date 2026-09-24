from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

try:
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:  # pragma: no cover
    BaseModel = object  # type: ignore
    ConfigDict = dict  # type: ignore
    def Field(default=None, **_: Any): return default


def now() -> datetime:
    return datetime.now(timezone.utc)


class Stage(str, Enum):
    SCOPING = "Scoping"
    PLANNING = "Planning"
    RESEARCHING = "Researching"
    SYNTHESIZING = "Synthesizing"
    VERIFYING = "Verifying"
    AWAITING_APPROVAL = "Awaiting Approval"
    PUBLISHED = "Published"
    FAILED = "Failed"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class TodoStatus(str, Enum):
    PLANNED = "planned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class IssueType(str, Enum):
    DRAFT_DEFECT = "DRAFT_DEFECT"
    EVIDENCE_GAP = "EVIDENCE_GAP"
    PLAN_GAP = "PLAN_GAP"


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScopedRequest(Model):
    objective: str
    research_boundaries: str
    constraints: str
    success_criteria: str
    deliverable_format: str
    clarification_turn_ids: list[str] = Field(default_factory=list)
    scope_hash: str = ""


class ResearchTodo(Model):
    todo_id: str = Field(default_factory=lambda: str(uuid4()))
    research_question: str
    status: TodoStatus = TodoStatus.PLANNED


class ArtifactRef(Model):
    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    kind: str
    path: str
    content_hash: str
    version: int = 1
    source_ids: list[str] = Field(default_factory=list)
    task_id: str | None = None


class SourceRecord(Model):
    source_id: str = Field(default_factory=lambda: str(uuid4()))
    locator: str
    retrieved_at: datetime = Field(default_factory=now)
    mcp_server: str = "controlled"
    mcp_tool: str = "search"
    provenance: dict[str, Any] = Field(default_factory=dict)


class IndexedChunk(Model):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    artifact_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationResult(Model):
    artifact_hash: str
    passed: bool
    checks: list[str] = Field(default_factory=list)
    issue_type: IssueType | None = None
    affected_ids: list[str] = Field(default_factory=list)
    repair_details: str = ""


class ApprovalDecision(Model):
    approval_id: str = Field(default_factory=lambda: str(uuid4()))
    artifact_hash: str
    verification_id: str
    approved: bool
    actor: str
    decided_at: datetime = Field(default_factory=now)


class PublicationRecord(Model):
    destination: str
    artifact_hash: str
    approval_id: str
    published: bool
    error: str = ""
    published_at: datetime | None = None


class ProgressAssessment(Model):
    completed_todos: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    ready_for_synthesis: bool = False
    next_action: str = ""


class ResearchGraphState(Model):
    run_id: str
    thread_id: str
    stage: Stage = Stage.SCOPING
    status: str = "active"
    input_turns: list[str] = Field(default_factory=list)
    scoped_request: ScopedRequest | None = None
    scope_questions: list[str] = Field(default_factory=list)
    todo_ids: list[str] = Field(default_factory=list)
    todos: list[ResearchTodo] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    evidence: list[ArtifactRef] = Field(default_factory=list)
    index_refs: list[str] = Field(default_factory=list)
    progress_assessment: ProgressAssessment | None = None
    draft: str = ""
    previous_draft: str = ""
    verification: VerificationResult | None = None
    recovery_attempt: int = 0
    recovery_budget: int = 3
    progress_fingerprint: str = ""
    approval: ApprovalDecision | None = None
    publication: PublicationRecord | None = None
    last_error: str = ""
