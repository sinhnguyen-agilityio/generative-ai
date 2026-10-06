"""What the UI derives from a run snapshot and its checkpoint, without Streamlit.

``RunView`` answers the questions every component asks — is a plan or a report waiting for the
user, is the run finished, which agent is doing what — so components only render.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from intelligence_collector.contracts import Stage, VerificationResult
from intelligence_collector.publication import can_request_approval

# (label shown on the card, agent name recorded in the activity log), in pipeline order.
AGENTS = (
    ("Scope Agent", "Scope Agent"),
    ("Planner", "Planner"),
    ("Researcher", "Research Supervisor"),
    ("Synthesizer", "Report Writer"),
    ("Verifier", "Verifier"),
)

# A finished thread takes no more input. Clarification, approval and recovery are durable pauses.
THREAD_DONE_STAGES = frozenset(
    {
        Stage.PUBLISHED.value,
        Stage.INCOMPLETE.value,
        Stage.CANCELLED.value,
        Stage.EXPORTED_INCOMPLETE.value,
    }
)
RESEARCH_DONE_STAGES = frozenset(
    {
        Stage.SYNTHESIZING.value,
        Stage.VERIFYING.value,
        Stage.AWAITING_APPROVAL.value,
        Stage.PUBLISHED.value,
        Stage.NEEDS_REVIEW.value,
    }
)
# Activity status of a node whose chat turn was interrupted before the node finished.
STOPPED = "stopped"
STOPPED_PROGRESS = "Stopped before finishing — use Continue unfinished run to resume"

PROGRESS_BY_STAGE = {
    Stage.SCOPING.value: "Waiting for your response",
    Stage.AWAITING_APPROVAL.value: "Report ready — reply in the chat to approve or request changes",
    Stage.NEEDS_REVIEW.value: "Needs human review",
    Stage.PUBLISHED.value: "Published",
}


def record_value(record: Any, key: str, default: Any = None) -> Any:
    """A field of a state record, whether it is a dict or a model."""
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def status_text(value: Any) -> str:
    return str(getattr(value, "value", value)).replace("_", " ")


def failure_stage_label(activity: list) -> str:
    """Name the node that was actually running when a turn-wide exception was caught, instead of
    always blaming whichever node happens to be first in that turn."""
    if not activity:
        return "Request"
    return activity[-1].get("label") or "Request"


def snapshot_verification(snapshot: dict) -> VerificationResult | None:
    value = snapshot.get("verification")
    if value is None or isinstance(value, VerificationResult):
        return value
    try:
        return VerificationResult.model_validate(value)
    except (TypeError, ValueError):
        return None


def current_report(snapshot: dict) -> str:
    return snapshot.get("report") or snapshot.get("latest_report") or snapshot.get("draft") or ""


def task_numbers(snapshot: dict) -> dict[str, int]:
    """Plan position of each todo, so logs can say "Task 2" instead of an id."""
    return {
        record_value(todo, "todo_id", ""): index + 1
        for index, todo in enumerate(snapshot.get("todos") or [])
    }


def agent_status(snapshot: dict, agent: str) -> str:
    """``pending``, ``running``, ``stopped``, ``failed`` or ``complete`` from the activity log
    (the latest entry of a repeated node decides). The Scope Agent is complete only once a
    research brief exists: asking a clarifying question is not finishing."""
    activity = list(snapshot.get("activity") or [])
    statuses = [
        record_value(item, "status") for item in activity if record_value(item, "agent") == agent
    ]
    if not statuses:
        status = "pending"
    elif "running" in statuses:
        status = "running"
    elif statuses[-1] == STOPPED:
        status = "stopped"
    elif "failure" in statuses:
        status = "failed"
    else:
        status = "complete"
    if agent == "Scope Agent":
        if snapshot.get("scoped_request"):
            return "complete"
        return "running" if status in {"running", "complete"} else status
    return status


@dataclass(frozen=True)
class RunView:
    """The run as the UI sees it in one rerun."""

    stage: str  # the stage shown in the header (the session's view of the run)
    snapshot: dict
    planning: bool  # a plan waits for approval
    reviewing: bool  # a verified report waits for approval
    clarifying: bool  # the Scope Agent asked a question
    recovering: bool  # the run needs a human recovery decision
    pending: bool  # the checkpoint has unfinished work to continue
    busy: bool = False  # a graph turn runs in this rerun: every input is disabled

    @classmethod
    def build(cls, stage: str, snapshot: dict, checkpoint, *, busy: bool = False) -> RunView:
        assessment = snapshot.get("scope_assessment") or {}
        verification = snapshot_verification(snapshot)
        return cls(
            stage=stage,
            snapshot=snapshot,
            planning=checkpoint is not None and checkpoint.interrupt_kind == "plan_approval",
            reviewing=snapshot.get("stage") == Stage.AWAITING_APPROVAL.value
            and verification is not None
            and can_request_approval(verification),
            clarifying=snapshot.get("stage") == Stage.SCOPING.value
            and assessment.get("need_clarification") is True
            and bool(str(assessment.get("question") or "").strip()),
            recovering=snapshot.get("stage") == Stage.NEEDS_REVIEW.value,
            pending=bool(
                checkpoint is not None
                and checkpoint.next_nodes
                and not checkpoint.has_interrupt
                and not checkpoint.terminal
            ),
            busy=busy,
        )

    @property
    def run_stage(self) -> str:
        """The stage stored in the run itself."""
        return self.snapshot.get("stage") or ""

    @property
    def waiting_for_you(self) -> bool:
        return self.planning or self.reviewing or self.clarifying or self.recovering

    @property
    def has_request(self) -> bool:
        return bool(self.snapshot.get("messages") or self.snapshot.get("request"))

    @property
    def finished(self) -> bool:
        return self.run_stage in THREAD_DONE_STAGES

    @property
    def input_disabled(self) -> bool:
        return self.busy or self.recovering or self.pending or self.finished

    @property
    def input_placeholder(self) -> str:
        if self.busy:
            return "Working on your reply…"
        if self.reviewing:
            return "What should change in the report?"
        if self.planning:
            return "What should change in the plan?"
        if self.clarifying:
            return "Answer the clarification question"
        if self.recovering:
            return "Use the recovery options above"
        return "What should we research?"
