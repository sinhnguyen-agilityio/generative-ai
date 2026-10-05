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
    INCOMPLETE = "Incomplete"


class WorkflowStatus(str, Enum):
    """Durable execution status, distinct from the node currently shown as ``Stage``.

    Waiting statuses identify the kind of validated human input required to resume a
    checkpoint. Terminal statuses make it explicit that an incomplete export or cancellation
    is not a successful publication.
    """

    RUNNING = "running"
    WAITING_FOR_CLARIFICATION = "waiting_for_clarification"
    WAITING_FOR_PLAN_APPROVAL = "waiting_for_plan_approval"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    WAITING_FOR_RECOVERY = "waiting_for_recovery"
    PUBLISHED = "published"
    EXPORTED_INCOMPLETE = "exported_incomplete"
    INCOMPLETE = "incomplete"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RecoveryAction(str, Enum):
    """Human-authorized transitions available at a recovery checkpoint.

    These are domain actions rather than graph node names. A later graph phase owns the fixed
    mapping from each action to its legal destination, preventing callers from selecting an
    arbitrary node.
    """

    RETRY_REPORT = "retry_report"
    REVISE = "revise"
    RESEARCH = "research"
    RESCOPE = "rescope"
    EXPORT_INCOMPLETE = "export_incomplete"
    CANCEL = "cancel"


class TodoStatus(str, Enum):
    """Lifecycle of one research todo."""

    PLANNED = "planned"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class Model(BaseModel):
    """Base for strict contracts: unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid")
    schema_version: int = 1


MAX_RECOVERY_FEEDBACK_LENGTH = 4_000
MAX_ADDITIONAL_RECOVERY_ATTEMPTS = 2


class RecoveryDecision(Model):
    """Validated human decision used to resume a ``NEEDS_REVIEW`` checkpoint.

    Revision, research, and scope changes require an instruction so the graph never invents
    the requested change. A retry must carry a fresh, explicitly bounded allowance. The
    contract deliberately contains no node or stage field: graph routing remains system-owned.
    """

    action: RecoveryAction
    feedback: str = Field(default="", max_length=MAX_RECOVERY_FEEDBACK_LENGTH)
    additional_attempts: int = Field(default=0, ge=0, le=MAX_ADDITIONAL_RECOVERY_ATTEMPTS)

    @model_validator(mode="after")
    def validate_action_requirements(self):
        self.feedback = self.feedback.strip()
        instruction_actions = {
            RecoveryAction.REVISE,
            RecoveryAction.RESEARCH,
            RecoveryAction.RESCOPE,
        }
        if self.action in instruction_actions and not self.feedback:
            raise ValueError(f"{self.action.value} requires non-empty feedback")
        if self.action is RecoveryAction.RETRY_REPORT:
            if self.additional_attempts == 0:
                raise ValueError("retry_report requires at least one additional attempt")
        elif self.additional_attempts != 0:
            raise ValueError("additional attempts are allowed only for retry_report")
        return self


class ApprovalResumeDecision(Model):
    """Validated response to the publication-approval interrupt."""

    approved: bool | None = None
    reply: str = Field(default="", max_length=MAX_RECOVERY_FEEDBACK_LENGTH)
    feedback: str = Field(default="", max_length=MAX_RECOVERY_FEEDBACK_LENGTH)
    actor: str = Field(default="local-user", min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_decision(self):
        self.reply = self.reply.strip()
        self.feedback = self.feedback.strip()
        self.actor = self.actor.strip()
        if self.approved is True and (self.reply or self.feedback):
            raise ValueError("approval cannot also contain revision feedback")
        if self.approved is False and not (self.reply or self.feedback):
            raise ValueError("rejection requires feedback")
        if self.approved is None and not (self.reply or self.feedback):
            raise ValueError("review response requires approval or a non-empty reply")
        return self


PLAN_APPROVAL_WORDS = frozenset(
    {
        "approve",
        "approved",
        "i approve",
        "yes",
        "ok",
        "okay",
        "go",
        "go ahead",
        "start",
        "looks good",
        "lgtm",
        "đồng ý",
        "duyệt",
        "phê duyệt",
        "ok, bắt đầu",
    }
)


class PlanApprovalDecision(Model):
    """Validated response to the plan-approval interrupt (initial or replacement plan).

    ``approved`` decides explicitly; otherwise ``reply`` is read: an approval phrase approves,
    "cancel"/"decline" declines, and any other text is a requested plan change.
    """

    approved: bool | None = None
    reply: str = Field(default="", max_length=MAX_RECOVERY_FEEDBACK_LENGTH)
    actor: str = Field(default="local-user", min_length=1, max_length=200)

    @model_validator(mode="after")
    def validate_decision(self):
        self.reply = self.reply.strip()
        self.actor = self.actor.strip()
        if self.approved is None and not self.reply:
            raise ValueError("plan decision requires approval or a non-empty reply")
        if self.approved is True and self.reply:
            raise ValueError("plan approval cannot also contain change requests")
        return self

    @property
    def outcome(self) -> Literal["approve", "decline", "change"]:
        if self.approved is not None:
            return (
                "approve"
                if self.approved and not self.reply
                else ("change" if self.reply else "decline")
            )
        text = self.reply.casefold().rstrip(".! ")
        if text in PLAN_APPROVAL_WORDS:
            return "approve"
        if text in {"cancel", "decline", "no", "stop", "hủy", "từ chối"}:
            return "decline"
        return "change"


class ClarificationResumeDecision(Model):
    """Validated response to a scope-clarification interrupt."""

    reply: str = Field(min_length=1, max_length=MAX_RECOVERY_FEEDBACK_LENGTH)
    actor: str = Field(default="local-user", min_length=1, max_length=200)

    @model_validator(mode="after")
    def normalize_fields(self):
        self.reply = self.reply.strip()
        self.actor = self.actor.strip()
        if not self.reply:
            raise ValueError("clarification reply must not be blank")
        if not self.actor:
            raise ValueError("clarification actor must not be blank")
        return self


class RunRecord(Model):
    """Discoverable metadata for one durable LangGraph run.

    The registry contains only lifecycle metadata. The LangGraph checkpointer remains the
    authoritative source for workflow state and the VFS remains authoritative for artifacts.
    """

    run_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    created_at: datetime
    updated_at: datetime
    workflow_status: WorkflowStatus
    current_stage: str = ""
    checkpoint_namespace: str = ""
    request_summary: str = ""
    terminal: bool = False


class RecoverySnapshot(Model):
    """Safe, UI-facing view of a registered run and its authoritative checkpoint."""

    run_id: str = Field(min_length=1)
    thread_id: str = Field(min_length=1)
    current_stage: str = ""
    workflow_status: WorkflowStatus
    has_interrupt: bool = False
    interrupt_kind: str = ""
    next_nodes: list[str] = Field(default_factory=list)
    last_successful_node: str = ""
    last_error: str = ""
    updated_at: datetime
    terminal: bool = False


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


MAX_REQUIRED_ANSWERS = 8


def _clean_items(items: list[str]) -> list[str]:
    return list(dict.fromkeys(item.strip() for item in items if item.strip()))


class ResearchBrief(Model):
    """The research question the whole run answers, written from the scope conversation.

    ``required_answers`` is the acceptance checklist: each answer the user explicitly asked for,
    in the user's terms. ``output_requirements`` lists only explicitly requested language,
    length and format constraints. ``original_question`` is system-owned: the user's own turns,
    verbatim, so later stages can check that nothing was added or lost.
    """

    research_question: str = Field(min_length=20)
    required_answers: list[str] = Field(default_factory=list, max_length=MAX_REQUIRED_ANSWERS)
    output_requirements: list[str] = Field(default_factory=list, max_length=MAX_REQUIRED_ANSWERS)
    original_question: str = ""

    @model_validator(mode="after")
    def reject_blank_fields(self):
        if not self.research_question.strip():
            raise ValueError("research brief question must not be blank")
        self.required_answers = _clean_items(self.required_answers)
        self.output_requirements = _clean_items(self.output_requirements)
        return self

    @property
    def checklist(self) -> list[str]:
        """Required answers; a brief without them is answered as one question."""
        return self.required_answers or [self.research_question]


class ResearchTodo(Model):
    """One planned research question and its status."""

    todo_id: str = Field(default_factory=lambda: str(uuid4()))
    research_question: str = Field(min_length=1)
    status: TodoStatus = TodoStatus.PLANNED
    research_attempt: int = Field(default=1, ge=1)
    research_instruction: str = ""
    evidence_needed: str = ""

    @model_validator(mode="after")
    def validate_question(self):
        if not self.research_question.strip():
            raise ValueError("research todo question must not be blank")
        return self


MAX_PLANNED_QUESTIONS = 5


class TodoPlan(Model):
    """Planner LLM output: the research questions, the evidence each needs (same order) and the
    expected report shape. Ids and status are system-owned."""

    questions: list[str] = Field(min_length=1, max_length=MAX_PLANNED_QUESTIONS)
    evidence_needed: list[str] = Field(default_factory=list, max_length=MAX_PLANNED_QUESTIONS)
    report_shape: str = ""

    @model_validator(mode="after")
    def validate_questions(self):
        self.questions = [question.strip() for question in self.questions]
        if any(not question for question in self.questions):
            raise ValueError("planned research questions must not be blank")
        if len({question.casefold() for question in self.questions}) != len(self.questions):
            raise ValueError("planned research questions must not repeat")
        self.evidence_needed = [item.strip() for item in self.evidence_needed][
            : len(self.questions)
        ]
        self.report_shape = self.report_shape.strip()
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


MAX_KEY_EXCERPTS = 5


class PageSummary(Model):
    """Page summarizer output: what one fetched page says about the research topic, plus up to
    five verbatim quotes. The researcher reads this instead of the page's raw text."""

    summary: str = Field(min_length=1)
    key_excerpts: list[str] = Field(default_factory=list, max_length=MAX_KEY_EXCERPTS)

    @model_validator(mode="after")
    def validate_summary(self):
        if not self.summary.strip():
            raise ValueError("page summary must not be blank")
        self.key_excerpts = [quote.strip() for quote in self.key_excerpts if quote.strip()]
        return self


class ResearchEvidenceCheck(Model):
    todo_id: str
    sufficient: bool
    missing_evidence: str = ""


class ResearchReadiness(str, Enum):
    READY = "READY"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


class ResearchEvidenceVerdict(Model):
    """Enough evidence for the user's goal, not perfection of every planned question."""

    status: ResearchReadiness
    questions: list[ResearchEvidenceCheck] = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_readiness(self):
        if self.status != ResearchReadiness.READY and not any(
            text.strip() for text in self.limitations
        ):
            raise ValueError("Partial or insufficient research requires explicit limitations")
        if self.status == ResearchReadiness.INSUFFICIENT and not any(
            not check.sufficient and check.missing_evidence.strip() for check in self.questions
        ):
            raise ValueError("Insufficient research requires a targeted gap instruction")
        return self


class DraftAssessment(Model):
    """Report writer output: the report as Markdown."""

    report_markdown: str = Field(min_length=1)


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


VerificationGate = Literal["coverage", "citation", "faithfulness", ""]
IssueType = Literal["draft_defect", "evidence_gap", "presentation"]


class VerificationCheck(Model):
    """One judged criterion: whether it passed and, when it failed, why.

    ``gate`` is one of the three verification gates. For a failed coverage item, ``item`` is
    the checklist text and ``missing_points`` the answer content the report lacks;
    ``issue_type`` drives repair diagnosis.
    """

    name: str
    passed: bool
    details: list[str] = Field(default_factory=list)
    affected_ids: list[str] = Field(default_factory=list)
    gate: VerificationGate = ""
    issue_type: IssueType = "draft_defect"
    item: str = ""
    missing_points: list[str] = Field(default_factory=list)


class VerificationResult(Model):
    """Outcome of the Verify step for one exact draft, bound to it by ``artifact_hash``."""

    verification_id: str = Field(default_factory=lambda: str(uuid4()))
    artifact_hash: str
    passed: bool
    checks: list[str] = Field(default_factory=list)
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
    """One coverage verdict: met or not, why, and for an unmet item the missing answer points.

    ``issue_type``: ``draft_defect`` when the evidence holds the answer but the report does not
    give it, ``evidence_gap`` when the evidence lacks it, ``presentation`` for a language,
    length, format or layout requirement.
    """

    criterion: str = Field(min_length=1)
    met: bool
    justification: str = Field(min_length=1)
    issue_type: IssueType = "draft_defect"
    missing_points: list[str] = Field(default_factory=list)


class ReportJudgeVerdict(Model):
    """Semantic verdict; structural citation and reviewer checks are appended separately."""

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
    Decides whether the reviewer's chat reply approves the report ("approve": publish it), asks
    for a change the existing evidence can support ("revise": rewrite and verify again), adds a
    section within the same question ("extend": stored evidence first, research only what is
    missing), or needs different evidence ("scope": back to scope clarification, the brief,
    planning and research). ``topic`` is the focused question of an "extend" request."""

    route: Literal["approve", "revise", "extend", "scope"]
    reason: str = ""
    topic: str = ""


class ExtensionEvidenceVerdict(Model):
    """Structured output of the extension judge (see agents/extension_judge.py): whether the
    passages already stored for this run are enough to write the section a reviewer asked for."""

    sufficient: bool
    relevant_source_ids: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    summary: str = ""
