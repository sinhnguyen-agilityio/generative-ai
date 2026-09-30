"""Pydantic contracts shared across the workflow: stages, LLM structured outputs, artifacts,
verification, approval and publication records.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def now() -> datetime:
    """Current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class Stage(str, Enum):
    """Workflow stage shown in the UI and stored in the graph state."""

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
    """Lifecycle of one research todo."""

    PLANNED = "planned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class IssueType(str, Enum):
    """Why a verification failed: ``DRAFT_DEFECT`` (a rewrite fixes it) or ``EVIDENCE_GAP`` (more
    research is needed).
    """

    DRAFT_DEFECT = "DRAFT_DEFECT"
    EVIDENCE_GAP = "EVIDENCE_GAP"


class Model(BaseModel):
    """Base for strict contracts: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1


class ScopeAssessment(BaseModel):
    """Scope Agent decision: ask one clarifying question, or confirm the scope with a verification
    message, never both.
    """

    model_config = ConfigDict(extra="forbid")
    need_clarification: bool
    question: str = ""
    verification: str = ""

    @model_validator(mode="after")
    def validate_decision(self):
        if self.need_clarification:
            if not self.question.strip():
                raise ValueError("clarification requires exactly one non-empty question")
            if self.verification.strip():
                raise ValueError("clarification cannot contain a verification message")
        else:
            if self.question.strip():
                raise ValueError("clear scope cannot contain a clarification question")
            if not self.verification.strip():
                raise ValueError("clear scope requires a non-empty verification message")
        return self


class ResearchBrief(Model):
    """The research question the whole run answers, written from the scope conversation."""

    research_question: str = Field(min_length=20)

    @model_validator(mode="after")
    def reject_blank_fields(self):
        if not self.research_question.strip():
            raise ValueError("research brief question must not be blank")
        return self


class ResearchTodo(Model):
    """One planned research question and its status."""

    todo_id: str = Field(default_factory=lambda: str(uuid4()))
    research_question: str = Field(min_length=1)
    status: TodoStatus = TodoStatus.PLANNED

    @model_validator(mode="after")
    def validate_question(self):
        if not self.research_question.strip():
            raise ValueError("research todo question must not be blank")
        return self


MAX_PLANNED_QUESTIONS = 5


class TodoPlan(Model):
    """Planner LLM output: the research questions. Ids and status are system-owned."""

    questions: list[str] = Field(min_length=1, max_length=MAX_PLANNED_QUESTIONS)

    @model_validator(mode="after")
    def validate_questions(self):
        self.questions = [question.strip() for question in self.questions]
        if any(not question for question in self.questions):
            raise ValueError("planned research questions must not be blank")
        if len({question.casefold() for question in self.questions}) != len(self.questions):
            raise ValueError("planned research questions must not repeat")
        return self


class ResearchFindings(Model):
    """Compressed output of one research sub-agent for one research question."""

    summary: str = Field(min_length=1)
    sufficient: bool
    unresolved_gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_findings(self):
        if not self.summary.strip():
            raise ValueError("research findings summary must not be blank")
        self.unresolved_gaps = [gap.strip() for gap in self.unresolved_gaps if gap.strip()]
        if not self.sufficient and not self.unresolved_gaps:
            raise ValueError("insufficient research must name at least one unresolved gap")
        return self


class DraftAssessment(Model):
    """Report writer output: the report as Markdown."""

    report_markdown: str = Field(min_length=1)
    claim_evidence_ids: dict[str, list[str]] = Field(default_factory=dict)


class ArtifactRef(Model):
    """Pointer to one file a run wrote to the VFS, with its content hash."""

    artifact_id: str = Field(default_factory=lambda: str(uuid4()))
    run_id: str
    kind: str
    path: str
    content_hash: str
    version: int = 1
    source_ids: list[str] = Field(default_factory=list)
    task_id: str | None = None


class SourceRecord(Model):
    """One captured web source and where it came from (MCP server and tool, provenance)."""

    source_id: str = Field(default_factory=lambda: str(uuid4()))
    locator: str
    retrieved_at: datetime = Field(default_factory=now)
    mcp_server: str = "controlled"
    mcp_tool: str = "search"
    provenance: dict[str, Any] = Field(default_factory=dict)


class IndexedChunk(Model):
    """A piece of source text stored in the vector index, with provenance metadata."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    source_id: str
    artifact_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationCheck(Model):
    """One programmatic source-verification check and, when it fails, why."""

    name: str
    passed: bool
    issue_type: IssueType | None = None
    details: list[str] = Field(default_factory=list)
    affected_ids: list[str] = Field(default_factory=list)


class VerificationResult(Model):
    """Outcome of the Verify step for one exact draft, bound to it by ``artifact_hash``."""

    verification_id: str = Field(default_factory=lambda: str(uuid4()))
    artifact_hash: str
    passed: bool
    checks: list[str] = Field(default_factory=list)
    issue_type: IssueType | None = None
    affected_ids: list[str] = Field(default_factory=list)
    repair_details: str = ""
    check_results: list[VerificationCheck] = Field(default_factory=list)


class Reflection(Model):
    """Self-improvement step of the BVF loop: learn from a failed verification before retrying."""

    diagnosis: str = Field(min_length=1)
    lessons: list[str] = Field(min_length=1, max_length=8)
    strategy: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_reflection(self):
        self.lessons = [lesson.strip() for lesson in self.lessons if lesson.strip()]
        if not self.lessons or not self.diagnosis.strip() or not self.strategy.strip():
            raise ValueError("reflection requires a diagnosis, lessons, and a strategy")
        return self


class ReportJudgeCriterion(Model):
    """One report-judge criterion verdict.

    ``issue_type`` lets the judge itself decide, per criterion, whether an unmet criterion is a
    wording/interpretation problem the writer can fix by revising (``DRAFT_DEFECT``) or a sign
    the gathered evidence itself is insufficient (``EVIDENCE_GAP``, routes back to targeted
    research instead of a rewrite). Defaults to ``DRAFT_DEFECT``, the far more common case.
    """

    criterion: str = Field(min_length=1)
    met: bool
    issue_type: IssueType = IssueType.DRAFT_DEFECT
    justification: str = Field(min_length=1)


class ReportJudgeVerdict(Model):
    """LLM-as-judge verdict for the whole report (see ``agents/report_judge.py``): goal
    achievement, per-question coverage, fair representation of the evidence, uncertainty
    disclosure, plain language, and safe content. This is the sole gate for the Verify step of
    Build-Verify-Fix — there is no separate deterministic citation/grounding script.
    """

    results: list[ReportJudgeCriterion] = Field(min_length=1)


class ApprovalDecision(Model):
    """A reviewer's decision, bound to the exact text and the verification it approves."""

    approval_id: str = Field(default_factory=lambda: str(uuid4()))
    artifact_hash: str
    verification_id: str
    approved: bool
    actor: str
    decided_at: datetime = Field(default_factory=now)


class PublicationRecord(Model):
    """Result of a publish attempt: where the report went, or why it was refused."""

    destination: str
    artifact_hash: str
    approval_id: str
    published: bool
    error: str = ""
    published_at: datetime | None = None


class FeedbackDecision(Model):
    """Structured output of the reviewer-reply classifier (see agents/feedback_router.py).
    Decides whether the reviewer's chat reply approves the report ("approve": publish it), only
    asks for a formatting change ("format": the writer re-presents the existing text) or asks for
    a change in what the report covers ("scope": back to scope clarification, the brief and the
    planner)."""

    route: Literal["approve", "format", "scope"]
    reason: str = ""


class ProgressAssessment(Model):
    """Supervisor check of whether research is ready for synthesis."""

    completed_todos: list[str] = Field(default_factory=list)
    blockers: list[str] = Field(default_factory=list)
    ready_for_synthesis: bool = False
    next_action: str = ""
