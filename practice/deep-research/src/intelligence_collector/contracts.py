from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

try:
    from pydantic import BaseModel, ConfigDict, Field, model_validator
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
    schema_version: int = 1


class ScopedRequest(Model):
    objective: str
    research_boundaries: str
    constraints: str
    success_criteria: str
    deliverable_format: str
    clarification_turn_ids: list[str] = Field(default_factory=list)
    scope_hash: str = ""


class ScopeAssessment(BaseModel):
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
    research_question: str = Field(min_length=20)

    @model_validator(mode="after")
    def reject_blank_fields(self):
        if not self.research_question.strip():
            raise ValueError("research brief question must not be blank")
        return self


class ResearchTodo(Model):
    todo_id: str = Field(default_factory=lambda: str(uuid4()))
    research_question: str = Field(min_length=1)
    status: TodoStatus = TodoStatus.PLANNED

    @model_validator(mode="after")
    def validate_question(self):
        if not self.research_question.strip():
            raise ValueError("research todo question must not be blank")
        return self


MAX_PLANNED_QUESTIONS = 5
MAX_ACCEPTANCE_CRITERIA = 5


class TodoPlan(Model):
    """Planner LLM output: research questions plus the acceptance criteria that define when the
    final report achieves the research brief's goal. Ids and status are system-owned."""

    questions: list[str] = Field(min_length=1, max_length=MAX_PLANNED_QUESTIONS)
    acceptance_criteria: list[str] = Field(min_length=1, max_length=MAX_ACCEPTANCE_CRITERIA)

    @model_validator(mode="after")
    def validate_questions(self):
        self.questions = [question.strip() for question in self.questions]
        if any(not question for question in self.questions):
            raise ValueError("planned research questions must not be blank")
        if len({question.casefold() for question in self.questions}) != len(self.questions):
            raise ValueError("planned research questions must not repeat")
        self.acceptance_criteria = [criterion.strip() for criterion in self.acceptance_criteria]
        if any(not criterion for criterion in self.acceptance_criteria):
            raise ValueError("acceptance criteria must not be blank")
        if len({c.casefold() for c in self.acceptance_criteria}) != len(self.acceptance_criteria):
            raise ValueError("acceptance criteria must not repeat")
        return self


class WorkerSearchPlan(Model):
    queries: list[str] = Field(min_length=1, max_length=5)
    stop_condition: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_search_plan(self):
        if any(not query.strip() for query in self.queries):
            raise ValueError("worker queries must not be blank")
        if not self.stop_condition.strip():
            raise ValueError("worker stop condition must not be blank")
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


class VerificationAssessment(Model):
    passed: bool
    issue_type: IssueType | None = None
    affected_claims: list[str] = Field(default_factory=list)
    feedback: str = ""

    @model_validator(mode="after")
    def validate_outcome(self):
        if self.passed and self.issue_type is not None:
            raise ValueError("passed verification cannot contain an issue type")
        if not self.passed and (self.issue_type is None or not self.feedback.strip()):
            raise ValueError("failed verification requires one issue type and feedback")
        return self


class SupervisorAssessment(Model):
    coverage_complete: bool
    research_complete: bool
    blockers: list[str] = Field(default_factory=list)
    delegation: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_completion(self):
        if self.research_complete and (not self.coverage_complete or self.blockers):
            raise ValueError("research completion requires coverage and no blockers")
        return self


class DraftAssessment(Model):
    report_markdown: str = Field(min_length=1)
    claim_evidence_ids: dict[str, list[str]] = Field(default_factory=dict)


class FixDecision(Model):
    action: str = Field(pattern="^(revise|research|replan|needs_review)$")
    reason: str = Field(min_length=1)
    target_ids: list[str] = Field(default_factory=list)


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


class VerificationCheck(Model):
    """One programmatic source-verification check and, when it fails, why."""

    name: str
    passed: bool
    issue_type: IssueType | None = None
    details: list[str] = Field(default_factory=list)
    affected_ids: list[str] = Field(default_factory=list)


class VerificationResult(Model):
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


class AcceptanceCriterionResult(Model):
    """One acceptance-criterion judgment: does the report actually achieve this goal?"""

    criterion: str = Field(min_length=1)
    met: bool
    justification: str = Field(min_length=1)


class AcceptanceVerdict(Model):
    """LLM judgment of whether the report meets every planner-defined acceptance criterion.

    This is a semantic, goal-level check ("does the report achieve what the brief asked for?"),
    distinct from the deterministic ``source_verification`` checks (citation syntax/grounding).
    """

    results: list[AcceptanceCriterionResult] = Field(min_length=1)


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


class FeedbackDecision(Model):
    """Structured output of the human-rejection-feedback classifier (see
    agents/feedback_router.py). Decides whether a reviewer's rejection reads as a
    wording/citation-level fix (route back to the report writer) or as a sign that the
    research itself went in the wrong direction (route back to the planner for a fresh set
    of research questions)."""

    route: Literal["writer", "planner"]
    reason: str = ""


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
    acceptance_criteria: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    evidence: list[ArtifactRef] = Field(default_factory=list)
    index_refs: list[str] = Field(default_factory=list)
    progress_assessment: ProgressAssessment | None = None
    draft: str = ""
    previous_draft: str = ""
    verification: VerificationResult | None = None
    verification_history: list[VerificationResult] = Field(default_factory=list)
    recovery_attempt: int = 0
    recovery_budget: int = 3
    progress_fingerprint: str = ""
    approval: ApprovalDecision | None = None
    publication: PublicationRecord | None = None
    last_error: str = ""
    human_feedback: str = ""
    feedback_route: str = ""
