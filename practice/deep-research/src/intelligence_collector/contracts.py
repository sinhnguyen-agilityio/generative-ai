"""Pydantic contracts shared across the workflow: stages, LLM structured outputs, artifacts,
verification, approval and publication records.
"""

from __future__ import annotations

import re
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
    CANCELLED = "Cancelled"
    EXPORTED_INCOMPLETE = "Exported Incomplete"


# RAGAS Faithfulness must be strictly greater than this for the third verification gate to pass.
# The workflow gate and the UI both read it from here.
FAITHFULNESS_THRESHOLD = 0.90


class RunStatus(str, Enum):
    """Coarse status of a run in the graph state (``GraphState.status``).

    The terminal values are also ``WorkflowStatus`` values, which the run registry records.
    """

    ACTIVE = "active"
    PAUSED = "paused"
    NEEDS_REVIEW = "needs_review"
    CANCELLED = "cancelled"
    EXPORTED_INCOMPLETE = "exported_incomplete"


class VerificationIssue(str, Enum):
    """Why the run left (or may leave) the happy path; ``GraphState.verification_issue``.

    ``NONE`` means no open issue. ``VERIFIER_FAILURE`` is an ERROR verdict (retried, never
    rewritten); research and publication failures go straight to human review.
    """

    NONE = ""
    DRAFT_DEFECT = "draft_defect"
    EVIDENCE_GAP = "evidence_gap"
    VERIFIER_FAILURE = "verifier_failure"
    RESEARCH_FAILURE = "research_failure"
    PUBLICATION_FAILURE = "publication_failure"


class RepairRoute(str, Enum):
    """The correction a failed verification needs (see ``repair.py``), largest first."""

    REPLAN = "replan"
    RESEARCH = "research"
    CHROMA = "chroma"
    VFS = "vfs"
    PRESENTATION = "presentation"


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


class PlanApprovalDecision(Model):
    """Validated response to the plan-approval interrupt (initial or replacement plan).

    Approve / Cancel are the plan card's buttons (``approved``); typed text (``reply``) is
    always a requested plan change.
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
        if self.reply:
            return "change"
        return "approve" if self.approved else "decline"


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


# ===== SUPERVISOR–SEARCHER RESEARCH (spec 002) =====


class PointStatus(str, Enum):
    """How far a searcher's evidence answers one evidence point."""

    ANSWERED = "answered"
    PARTIAL = "partial"
    MISSING = "missing"


class RejectReason(str, Enum):
    """Why a searcher did not keep a search result."""

    OFF_TOPIC = "off_topic"
    NO_USABLE_CONTENT = "no_usable_content"
    DUPLICATE = "duplicate"
    LOW_CREDIBILITY = "low_credibility"
    OUTSIDE_SCOPE = "outside_scope"


class StopReason(str, Enum):
    """Why a searcher stopped searching (the first condition that held)."""

    ALL_POINTS_COVERED = "all_points_covered"
    SEARCH_BUDGET_SPENT = "search_budget_spent"
    PAGE_BUDGET_FULL = "page_budget_full"
    NO_PROGRESS = "no_progress"
    SEARCH_FAILURES = "search_failures"
    MODEL_FINISHED = "model_finished"


class ReportStatus(str, Enum):
    """Whether a searcher report was used as written, corrected by code, or unreadable."""

    COMPLETE = "complete"
    CORRECTED = "corrected"
    UNREADABLE = "unreadable"


class FailureKind(str, Enum):
    """Why a task produced no usable evidence: a technical failure or nothing relevant found."""

    TECHNICAL = "technical"
    CONTENT = "content"


class TaskOrigin(str, Enum):
    """Whether a research task comes from the approved plan or from gap analysis."""

    PLAN = "plan"
    FOLLOW_UP = "follow_up"


class ClaimVerdict(str, Enum):
    """How far a report sentence's cited passages support it."""

    SUPPORTED = "supported"
    PARTLY_SUPPORTED = "partly_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"


MAX_EVIDENCE_POINTS = 4
MAX_EVIDENCE_POINT_CHARS = 300
_POINT_ID = re.compile(r"^E[1-4]$")


class EvidencePoint(Model):
    """One specific thing a research task must find, approved with the plan.

    ``required_answers`` holds the brief checklist items it serves; a point linked to none is
    non-essential: missing it never creates a follow-up task.
    """

    point_id: str
    text: str = Field(min_length=1, max_length=MAX_EVIDENCE_POINT_CHARS)
    required_answers: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_point(self):
        if not _POINT_ID.match(self.point_id):
            raise ValueError("evidence point ids are E1..E4")
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("evidence point text must not be blank")
        self.required_answers = _clean_items(self.required_answers)
        return self

    @property
    def essential(self) -> bool:
        return bool(self.required_answers)


class EvidencePointDraft(Model):
    """Planner (or supervisor follow-up) output for one evidence point: the text and the 1-based
    indexes of the required answers it serves."""

    text: str = Field(min_length=1, max_length=MAX_EVIDENCE_POINT_CHARS)
    required_answers: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_draft(self):
        self.text = self.text.strip()
        if not self.text:
            raise ValueError("evidence point text must not be blank")
        if any(index < 1 for index in self.required_answers):
            raise ValueError("required_answers are 1-based indexes")
        self.required_answers = list(dict.fromkeys(self.required_answers))
        return self


def evidence_points_from_drafts(
    drafts: list[EvidencePointDraft], checklist: list[str]
) -> list[EvidencePoint]:
    """Number the drafts E1.. and resolve their indexes to checklist texts. Raises ``ValueError``
    for an index outside the checklist."""
    points = []
    for number, draft in enumerate(drafts, start=1):
        bad = [i for i in draft.required_answers if i > len(checklist)]
        if bad:
            raise ValueError(
                f"required_answers {bad} are outside 1..{len(checklist)} for: {draft.text}"
            )
        points.append(
            EvidencePoint(
                point_id=f"E{number}",
                text=draft.text,
                required_answers=[checklist[i - 1] for i in draft.required_answers],
            )
        )
    return points



class ResearchTodo(Model):
    """One planned research question and its status."""

    todo_id: str = Field(default_factory=lambda: str(uuid4()))
    research_question: str = Field(min_length=1)
    status: TodoStatus = TodoStatus.PLANNED
    research_attempt: int = Field(default=1, ge=1)
    research_instruction: str = ""
    evidence_needed: str = ""
    # 1-4 for planned tasks; empty only on todos from checkpoints older than evidence points
    # (graph/migration.py derives one point for those).
    evidence_points: list[EvidencePoint] = Field(
        default_factory=list, max_length=MAX_EVIDENCE_POINTS
    )
    origin: TaskOrigin = TaskOrigin.PLAN
    round: int = Field(default=1, ge=1)
    # For a follow-up: the "todo_id:point_id" gaps that prompted it (it is not linked to them).
    gap_point_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_question(self):
        if not self.research_question.strip():
            raise ValueError("research todo question must not be blank")
        ids = [point.point_id for point in self.evidence_points]
        if len(ids) != len(set(ids)):
            raise ValueError("evidence point ids must be unique within a task")
        return self


MAX_PLANNED_QUESTIONS = 5


class TodoPlan(Model):
    """Planner LLM output: the research questions, the evidence each needs (same order) and the
    expected report shape. Ids and status are system-owned."""

    questions: list[str] = Field(min_length=1, max_length=MAX_PLANNED_QUESTIONS)
    evidence_needed: list[str] = Field(default_factory=list, max_length=MAX_PLANNED_QUESTIONS)
    # Per question, in the same order: 1-4 evidence points with the 1-based indexes of the
    # required answers each serves. Index ranges are checked against the brief by the planner.
    evidence_points: list[list[EvidencePointDraft]] = Field(
        min_length=1, max_length=MAX_PLANNED_QUESTIONS
    )
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
        if len(self.evidence_points) != len(self.questions):
            raise ValueError("evidence_points must give one list per question, in order")
        for points in self.evidence_points:
            if not 1 <= len(points) <= MAX_EVIDENCE_POINTS:
                raise ValueError(f"each question needs 1 to {MAX_EVIDENCE_POINTS} evidence points")
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


class ResearchReadiness(str, Enum):
    READY = "READY"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


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
    Approval is the "Approve and publish" button, never a typed reply. Decides whether the
    reviewer's chat reply asks for a change the existing evidence can support ("revise": rewrite
    and verify again), adds a
    section within the same question ("extend": stored evidence first, research only what is
    missing), or needs different evidence ("scope": back to scope clarification, the brief,
    planning and research). ``topic`` is the focused question of an "extend" request."""

    route: Literal["revise", "extend", "scope"]
    reason: str = ""
    topic: str = ""


class ExtensionEvidenceVerdict(Model):
    """Structured output of the extension judge (see agents/extension_judge.py): whether the
    passages already stored for this run are enough to write the section a reviewer asked for."""

    sufficient: bool
    relevant_source_ids: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    summary: str = ""


class ResultRowDecision(Model):
    """Result evaluator output for one search row (by its 1-based index in the batch)."""

    index: int = Field(ge=1)
    decision: Literal["accepted", "rejected"]
    reason: Literal["off_topic", "low_credibility", "outside_scope"] | None = None
    covers: list[str] = Field(default_factory=list)


class ResultEvaluationBatch(Model):
    """Structured output of the result evaluator for one search."""

    rows: list[ResultRowDecision] = Field(default_factory=list)


class ResultEvaluation(Model):
    """One search result's decision for one task (stored in ``evaluations.json``)."""

    url: str
    title: str = ""
    query: str = ""
    search_index: int = Field(ge=1)
    decision: Literal["accepted", "rejected"]
    reason: RejectReason | None = None
    covers: list[str] = Field(default_factory=list)
    linked_to_task: str | None = None
    snippet_only: bool = False

    @model_validator(mode="after")
    def validate_decision(self):
        if self.decision == "rejected":
            if self.reason is None:
                raise ValueError("a rejected result needs a reason")
            if self.covers:
                raise ValueError("a rejected result covers no evidence point")
            if self.snippet_only:
                raise ValueError("snippet_only applies to accepted results")
        elif self.reason is not None:
            raise ValueError("an accepted result has no rejection reason")
        return self


MAX_QUOTE_WORDS = 30
MAX_REPORT_FINDINGS = 5


class PointSupport(Model):
    """One source behind an answered or partial evidence point, with a verbatim quote."""

    source_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class PointReport(Model):
    """A searcher's status for one evidence point."""

    point_id: str
    status: PointStatus
    support: list[PointSupport] = Field(default_factory=list)
    lookup_used: bool = False

    @model_validator(mode="after")
    def validate_support(self):
        if self.status != PointStatus.MISSING and not self.support:
            raise ValueError("an answered or partial point needs at least one supporting quote")
        return self


class Finding(Model):
    """One key finding of a searcher report, citing accepted sources."""

    text: str = Field(min_length=1)
    source_ids: list[str] = Field(min_length=1)


class PointReportDraft(Model):
    """Searcher-report model output for one point (validated and corrected by code)."""

    point_id: str
    status: PointStatus
    support: list[PointSupport] = Field(default_factory=list)


class SearcherReportDraft(Model):
    """Structured output of the searcher-report call."""

    points: list[PointReportDraft] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)


class SearcherReport(Model):
    """The short structured result of one research task (contracts/searcher-report.md)."""

    todo_id: str
    points: list[PointReport]
    findings: list[Finding] = Field(default_factory=list, max_length=MAX_REPORT_FINDINGS)
    accepted_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    queries: list[str] = Field(default_factory=list)
    stop_reason: StopReason
    failure_kind: FailureKind | None = None
    report_status: ReportStatus = ReportStatus.COMPLETE
    corrections: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_report(self):
        ids = [point.point_id for point in self.points]
        if len(ids) != len(set(ids)):
            raise ValueError("one entry per evidence point")
        if self.report_status == ReportStatus.COMPLETE and self.corrections:
            raise ValueError("a corrected report must say so")
        return self


class PassageLookupRecord(Model):
    """One stored-passage lookup for one evidence point."""

    lookup_id: str = Field(default_factory=lambda: str(uuid4()))
    requested_by: str
    point_key: str
    queries: list[str] = Field(default_factory=list)
    allowed_source_ids: list[str] = Field(default_factory=list)
    results: list[dict[str, Any]] = Field(default_factory=list, max_length=20)
    fallback: Literal["none", "vector_only", "combined_query_only"] = "none"
    error: str = ""


class Gap(Model):
    """One missing or partial evidence point in a gap analysis."""

    point_key: str
    text: str
    status: PointStatus
    essential: bool
    required_answers: list[str] = Field(default_factory=list)
    resolution: Literal[
        "covered_by_report", "covered_by_lookup", "follow_up", "limitation", "open"
    ] = "open"
    covering_source_id: str = ""
    quote: str = ""
    lookup_id: str = ""
    justification: str = ""


class GapAnalysis(Model):
    """The supervisor's analysis at the end of one research round."""

    round: int = Field(ge=1)
    gaps: list[Gap] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    refusals: list[dict[str, Any]] = Field(default_factory=list)
    closed_essential: int = Field(default=0, ge=0)
    stall_count: int = Field(default=0, ge=0)
    turns_used: int = Field(default=0, ge=0, le=4)
    turn_limit_reached: bool = False
    ended: Literal[
        "no_essential_gaps", "caps_reached", "stalled", "turn_limit", "follow_ups_created"
    ] = "no_essential_gaps"


class Claim(Model):
    """One fact-stating sentence of a draft and its claim-level verdict."""

    claim_id: str
    section: str = ""
    sentence: str
    cited_source_ids: list[str] = Field(default_factory=list)
    states_fact: bool = True
    verdict: ClaimVerdict | None = None
    per_source: list[dict[str, Any]] = Field(default_factory=list)
    unsupported_parts: list[str] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


class ClaimEvaluation(Model):
    """Claim-level faithfulness and citation support of one draft (the faithfulness gate)."""

    draft_version: int = Field(ge=0)
    artifact_hash: str
    claims: list[Claim] = Field(default_factory=list)
    supported_share: float = Field(ge=0, le=1)
    passed: bool

class QueryPhrasings(Model):
    """Alternative phrasings for one evidence point."""

    point_id: str
    phrasings: list[str] = Field(default_factory=list, max_length=3)


class QueryRewrite(Model):
    """Structured output of the query rewriter (2-3 phrasings per evidence point)."""

    points: list[QueryPhrasings] = Field(default_factory=list)


class GapCoverageVerdict(Model):
    """The supervisor's judgment for one essential gap against looked-up passages."""

    point_key: str
    covered: bool
    source_id: str = ""
    quote: str = ""
    justification: str = ""


class GapCoverageBatch(Model):
    """Structured output of the gap coverage judgment."""

    gaps: list[GapCoverageVerdict] = Field(default_factory=list)


class SourceSupport(Model):
    """Whether one cited source supports part of a claim sentence."""

    source_id: str
    supports_part: bool


class ClaimJudgment(Model):
    """Claim judge output for one sentence."""

    claim_id: str
    states_fact: bool
    verdict: ClaimVerdict
    per_source: list[SourceSupport] = Field(default_factory=list)
    unsupported_parts: list[str] = Field(default_factory=list)


class ClaimJudgeBatch(Model):
    """Structured output of one claim judge call."""

    claims: list[ClaimJudgment] = Field(default_factory=list)
