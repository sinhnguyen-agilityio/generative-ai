"""Checkpoint inspection and controlled resume operations.

The run registry makes threads discoverable; LangGraph checkpoints remain authoritative for the
state and legal continuation. This service is deliberately independent of Streamlit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.types import Command
from intelligence_collector.graph.migration import normalize_checkpoint

from intelligence_collector.contracts import (
    ApprovalResumeDecision,
    ClarificationResumeDecision,
    PlanApprovalDecision,
    RecoveryDecision,
    RecoverySnapshot,
    RunRecord,
    WorkflowStatus,
)


class RecoveryServiceError(RuntimeError):
    """Base error for a refused or unavailable recovery operation."""


class RecoverableRunNotFound(RecoveryServiceError):
    """No run-registry entry exists for the requested thread."""


class CheckpointNotFound(RecoveryServiceError):
    """The registry entry has no authoritative LangGraph checkpoint."""


class CheckpointIdentityMismatch(RecoveryServiceError):
    """Registry and checkpoint identities disagree, so recovery is unsafe."""


class InvalidRecoveryOperation(RecoveryServiceError):
    """The requested continuation is incompatible with the checkpoint state."""


class WorkflowRecoveryService:
    """Discover, inspect, and resume registered LangGraph runs."""

    def __init__(self, graph, run_registry):
        self.graph = graph
        self.run_registry = run_registry

    def _config(self, thread_id: str) -> dict:
        record = self._registered(thread_id)
        return {"configurable": {"thread_id": thread_id,
                                 "checkpoint_ns": record.checkpoint_namespace},
                "recursion_limit": 100}

    def load_state(self, thread_id: str) -> dict:
        """Restore authoritative state without starting a new request or replaying START."""
        self.inspect(thread_id)
        checkpoint = self.graph.get_state(self._config(thread_id))
        state = normalize_checkpoint(dict(checkpoint.values))
        exhausted_gap = ("prepare_gap_research" in checkpoint.next and
                         state.get("research_gap_attempt", 0) >= state["research_gap_budget"])
        if "replan" in checkpoint.next or exhausted_gap:
            # Redirect obsolete plan repair or exhausted legacy gap work without replaying
            # START or resetting the consumed research allowance.
            todos = state.get("todos", [])
            if not todos:
                raise InvalidRecoveryOperation("Legacy replan checkpoint needs an explicit rescope")
            instruction = dict(checkpoint.values).get("plan_repair_instruction") or "Find evidence for the core research goal"
            state.update(verification=None, approval=None, publication=None,
                         research_status="INSUFFICIENT", result_incomplete=True,
                         research_limitations=[instruction], verification_issue="evidence_gap",
                         research_gaps=[{"todo_id": todos[0]["todo_id"], "instruction": instruction}],
                         progress_assessment={"ready_for_synthesis": False})
            self.graph.update_state(self._config(thread_id), state, as_node="research_complete")
            return self._clean_result(self.graph.invoke(None, self._config(thread_id)))
        # Older workflows ended at these human-waiting states. Bridge them through
        # existing graph edges rather than restarting scoping or bypassing approval.
        if not checkpoint.next and not self._interrupts(checkpoint):
            stage = state.get("stage")
            if stage == "Awaiting Approval":
                # Legacy ended approval checkpoints have no resumable human interrupt.
                # Invalidate stale authority and enter the controlled recovery path.
                state.update(stage="NEEDS_REVIEW", approval=None, verification=None,
                             verification_issue="publication_failure",
                             last_error="Legacy approval checkpoint requires re-verification")
                stage = "NEEDS_REVIEW"
            node = ("needs_review" if stage == "NEEDS_REVIEW" else
                    "clarify_with_user" if stage == "Scoping" and
                    (state.get("scope_assessment") or {}).get("need_clarification") else None)
            if node is not None:
                self.graph.update_state(self._config(thread_id), state, as_node=node)
                return self._clean_result(self.graph.invoke(None, self._config(thread_id)))
        return state

    def list_recoverable_runs(self) -> list[RunRecord]:
        return self.run_registry.list_unfinished()

    def discover_legacy_runs(self) -> list[RunRecord]:
        """Explicitly backfill the registry from existing root-graph checkpoints."""
        saver = self.graph.checkpointer
        if saver is None:
            raise CheckpointNotFound("No durable checkpointer is configured")
        seen = set()
        # SQLite saver iteration holds its connection lock. Collect before calling
        # get_state/inspect, which needs that same lock.
        entries = list(saver.list(None))
        for entry in entries:
            config = entry.config.get("configurable", {})
            thread_id = config.get("thread_id")
            if not thread_id or config.get("checkpoint_ns", "") or thread_id in seen:
                continue
            seen.add(thread_id)
            values = entry.checkpoint.get("channel_values", {})
            if not values.get("run_id"):
                continue
            normalize_checkpoint(values)
            if values.get("thread_id", thread_id) != thread_id:
                raise CheckpointIdentityMismatch("Legacy checkpoint thread identity mismatch")
            self.run_registry.register_run(values["run_id"], thread_id,
                                           request_summary=values.get("request", ""))
            snapshot = self.inspect(thread_id)
            self.run_registry.update_run(snapshot.run_id, thread_id,
                                         workflow_status=snapshot.workflow_status,
                                         current_stage=snapshot.current_stage)
        return self.list_recoverable_runs()

    def _registered(self, thread_id: str) -> RunRecord:
        thread_id = str(thread_id).strip()
        if not thread_id:
            raise ValueError("thread_id must be non-empty")
        record = self.run_registry.get_by_thread_id(thread_id)
        if record is None:
            raise RecoverableRunNotFound(
                f"No registered workflow run exists for thread '{thread_id}'"
            )
        return record

    @staticmethod
    def _interrupts(checkpoint) -> list[Any]:
        return [
            interrupt
            for task in (getattr(checkpoint, "tasks", ()) or ())
            for interrupt in (getattr(task, "interrupts", ()) or ())
        ]

    @staticmethod
    def _interrupt_kind(interrupts: list[Any]) -> str:
        if not interrupts:
            return ""
        value = getattr(interrupts[0], "value", None)
        if isinstance(value, Mapping):
            return str(value.get("kind") or "")
        return ""

    @staticmethod
    def _last_successful_node(checkpoint) -> str:
        metadata = getattr(checkpoint, "metadata", {}) or {}
        writes = metadata.get("writes") if isinstance(metadata, Mapping) else None
        if isinstance(writes, Mapping) and writes:
            return str(next(iter(writes)))
        return ""

    def inspect(self, thread_id: str) -> RecoverySnapshot:
        """Read the registry and authoritative checkpoint without mutating either."""
        record = self._registered(thread_id)
        checkpoint = self.graph.get_state(self._config(record.thread_id))
        values = dict(getattr(checkpoint, "values", {}) or {})
        if not values:
            raise CheckpointNotFound(
                f"No checkpoint state exists for thread '{record.thread_id}'"
            )
        normalize_checkpoint(values)
        checkpoint_run_id = str(values.get("run_id") or "")
        checkpoint_thread_id = str(values.get("thread_id") or record.thread_id)
        if checkpoint_run_id != record.run_id or checkpoint_thread_id != record.thread_id:
            raise CheckpointIdentityMismatch(
                "Run registry identity does not match the checkpoint state"
            )
        interrupts = self._interrupts(checkpoint)
        stage = str(values.get("stage") or record.current_stage)
        terminal_status = {"Published": WorkflowStatus.PUBLISHED,
                           "Incomplete": WorkflowStatus.INCOMPLETE,
                           "Cancelled": WorkflowStatus.CANCELLED,
                           "Exported Incomplete": WorkflowStatus.EXPORTED_INCOMPLETE}.get(stage)
        waiting_status = {"NEEDS_REVIEW": WorkflowStatus.WAITING_FOR_RECOVERY,
                          "Awaiting Approval": WorkflowStatus.WAITING_FOR_APPROVAL}.get(stage)
        if self._interrupt_kind(interrupts) == "plan_approval":
            waiting_status = WorkflowStatus.WAITING_FOR_PLAN_APPROVAL
        return RecoverySnapshot(
            run_id=record.run_id,
            thread_id=record.thread_id,
            current_stage=stage,
            workflow_status=terminal_status or waiting_status or record.workflow_status,
            has_interrupt=bool(interrupts),
            interrupt_kind=self._interrupt_kind(interrupts),
            next_nodes=[str(node) for node in (getattr(checkpoint, "next", ()) or ())],
            last_successful_node=self._last_successful_node(checkpoint),
            last_error=str(values.get("last_error") or ""),
            updated_at=record.updated_at,
            terminal=terminal_status is not None,
        )

    @staticmethod
    def _clean_result(result) -> dict:
        if not isinstance(result, Mapping):
            raise RecoveryServiceError("Resumed graph did not return workflow state")
        return {key: value for key, value in result.items() if key != "__interrupt__"}

    def continue_run(self, thread_id: str) -> dict:
        """Continue a non-interrupted unfinished checkpoint from its graph-owned next node."""
        snapshot = self.inspect(thread_id)
        if snapshot.terminal:
            raise InvalidRecoveryOperation("A terminal run cannot be continued")
        if snapshot.has_interrupt:
            raise InvalidRecoveryOperation(
                "Checkpoint is waiting for human input; use resume_interrupt"
            )
        if not snapshot.next_nodes:
            raise InvalidRecoveryOperation(
                "Checkpoint has no pending graph node to continue"
            )
        values = self.graph.get_state(self._config(snapshot.thread_id)).values
        exhausted_gap = ("prepare_gap_research" in snapshot.next_nodes and
                         values.get("research_gap_attempt", 0) >= min(values.get("research_gap_budget", 1), 1))
        if "replan" in snapshot.next_nodes or exhausted_gap:
            return self.load_state(thread_id)
        result = self.graph.invoke(None, self._config(snapshot.thread_id))
        return self._clean_result(result)

    @staticmethod
    def _validated_resume(kind: str, decision: Mapping[str, Any]) -> dict:
        if kind == "publication_approval":
            model = ApprovalResumeDecision.model_validate(decision)
        elif kind == "clarification_required":
            model = ClarificationResumeDecision.model_validate(decision)
        elif kind == "plan_approval":
            model = PlanApprovalDecision.model_validate(decision)
        elif kind == "recovery_required":
            model = RecoveryDecision.model_validate(decision)
        else:
            raise InvalidRecoveryOperation(
                f"Unsupported checkpoint interrupt kind: '{kind or 'unknown'}'"
            )
        return model.model_dump(mode="json", exclude={"schema_version"})

    def resume_interrupt(self, thread_id: str, decision: Mapping[str, Any]) -> dict:
        """Resume an interrupted checkpoint with input validated for its interrupt kind."""
        snapshot = self.inspect(thread_id)
        if snapshot.terminal:
            raise InvalidRecoveryOperation("A terminal run cannot be resumed")
        if not snapshot.has_interrupt:
            raise InvalidRecoveryOperation("Checkpoint is not waiting at an interrupt")
        if not isinstance(decision, Mapping):
            raise TypeError("resume decision must be a mapping")
        payload = self._validated_resume(snapshot.interrupt_kind, decision)
        result = self.graph.invoke(
            Command(resume=payload), self._config(snapshot.thread_id)
        )
        return self._clean_result(result)
