"""Checkpoint inspection and controlled resume operations.

The run registry makes threads discoverable; LangGraph checkpoints remain authoritative for the
state and legal continuation. This service is deliberately independent of Streamlit.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langgraph.types import Command

from intelligence_collector.contracts import (
    ApprovalResumeDecision,
    ClarificationResumeDecision,
    RecoveryDecision,
    RecoverySnapshot,
    RunRecord,
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

    @staticmethod
    def _config(thread_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": thread_id}}

    def list_recoverable_runs(self) -> list[RunRecord]:
        return self.run_registry.list_unfinished()

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
        checkpoint_run_id = str(values.get("run_id") or "")
        checkpoint_thread_id = str(values.get("thread_id") or record.thread_id)
        if checkpoint_run_id != record.run_id or checkpoint_thread_id != record.thread_id:
            raise CheckpointIdentityMismatch(
                "Run registry identity does not match the checkpoint state"
            )
        interrupts = self._interrupts(checkpoint)
        return RecoverySnapshot(
            run_id=record.run_id,
            thread_id=record.thread_id,
            current_stage=str(values.get("stage") or record.current_stage),
            workflow_status=record.workflow_status,
            has_interrupt=bool(interrupts),
            interrupt_kind=self._interrupt_kind(interrupts),
            next_nodes=[str(node) for node in (getattr(checkpoint, "next", ()) or ())],
            last_successful_node=self._last_successful_node(checkpoint),
            last_error=str(values.get("last_error") or ""),
            updated_at=record.updated_at,
            terminal=record.terminal,
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
        result = self.graph.invoke(None, self._config(snapshot.thread_id))
        return self._clean_result(result)

    @staticmethod
    def _validated_resume(kind: str, decision: Mapping[str, Any]) -> dict:
        if kind == "publication_approval":
            model = ApprovalResumeDecision.model_validate(decision)
        elif kind == "clarification_required":
            model = ClarificationResumeDecision.model_validate(decision)
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
