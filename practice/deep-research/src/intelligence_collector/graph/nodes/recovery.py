"""Applies a validated human recovery decision after ``needs_review``."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from intelligence_collector.contracts import (
    RecoveryAction,
    RecoveryDecision,
    ResearchTodo,
    RunStatus,
    Stage,
    TodoStatus,
)
from intelligence_collector.graph.nodes.common import (
    invalidate_draft_authority,
    reset_research_rounds,
)
from intelligence_collector.ports import ArtifactStore


class RecoveryNodes:
    """Apply human authorization before any recovery work runs."""

    def __init__(self, vfs: ArtifactStore):
        self.vfs = vfs

    def apply(self, state):
        decision = RecoveryDecision.model_validate(state.get("recovery_decision"))
        action = decision.action
        history = [
            *state.get("bvf_history", []),
            {
                "kind": "RECOVERY_DECISION",
                **decision.model_dump(mode="json"),
            },
        ]
        request = f"Recovery action: {action.value.replace('_', ' ')}" + (
            f" — {decision.feedback}" if decision.feedback else ""
        )
        update = {
            "recovery_decision": None,
            "recovery_route": action.value,
            "bvf_history": history,
            "approval": None,
            "publication": None,
            # The reviewer's decision is part of the chat, like a review reply.
            "messages": [*state.get("messages", []), HumanMessage(content=request)],
        }
        if action is RecoveryAction.CANCEL:
            return {
                **update,
                "stage": Stage.CANCELLED.value,
                "status": RunStatus.CANCELLED.value,
                "current_action": "Run cancelled by reviewer",
            }
        if action is RecoveryAction.EXPORT_INCOMPLETE:
            content = (
                "# INCOMPLETE — unverified research report\n\n"
                "This artifact has unresolved verification issues and is not published.\n\n"
                + state.get("last_error", "")
                + "\n\n"
                + state.get("draft", "")
            )
            ref = self.vfs.write(
                state["run_id"], "reports/incomplete.md", content, "incomplete_report"
            )
            return {
                **update,
                "stage": Stage.EXPORTED_INCOMPLETE.value,
                "status": RunStatus.EXPORTED_INCOMPLETE.value,
                "current_action": "Incomplete report saved for review",
                "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
            }
        # Each human authorization starts a fresh, small recovery allowance. Verification history
        # and artifacts remain available, but no stale verification/approval follows the new draft.
        update.update(
            invalidate_draft_authority(),
            status=RunStatus.ACTIVE.value,
            recovery_attempt=0,
            recovery_budget=decision.additional_attempts or 1,
            stall_count=0,
            progress_fingerprint="",
            last_error="",
            research_gap_attempt=0,
            research_gap_budget=1,
        )
        if action in {RecoveryAction.RETRY_REPORT, RecoveryAction.REVISE}:
            feedback = decision.feedback or "Repair the unresolved verification issues: " + (
                state.get("last_error")
                or (state.get("verification") or {}).get("repair_details", "")
            )
            return {
                **update,
                "stage": Stage.SYNTHESIZING.value,
                "human_feedback": feedback,
                "reviewer_requests": [*state.get("reviewer_requests", []), feedback],
            }
        if action is RecoveryAction.RESEARCH:
            todos = [ResearchTodo.model_validate(item) for item in state.get("todos", [])]
            affected = {gap.get("todo_id") for gap in state.get("research_gaps", [])}
            reopened = False
            for todo in todos:
                if todo.todo_id in affected:
                    todo.status = TodoStatus.PLANNED
                    todo.research_attempt += 1
                    todo.research_instruction = decision.feedback
                    reopened = True
            if not reopened:
                todos.append(ResearchTodo(research_question=decision.feedback))
            # Preserve completed plan coverage; only planned tasks are delegated.
            return {
                **update,
                "stage": Stage.RESEARCHING.value,
                "human_feedback": "",
                "todos": [todo.model_dump(mode="json") for todo in todos],
                "todo_ids": [todo.todo_id for todo in todos],
            }
        return {
            **update,
            **reset_research_rounds(),
            "stage": Stage.SCOPING.value,
            "request": decision.feedback,
            "draft": "",
            "human_feedback": "",
            "reviewer_requests": [],
        }
