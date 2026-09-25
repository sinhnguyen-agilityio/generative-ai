from __future__ import annotations

from intelligence_collector.agents.supervisor import assess
from intelligence_collector.agents.synthesis import synthesize
from intelligence_collector.agents.verification import verify
from intelligence_collector.agents.worker import run_worker
from intelligence_collector.contracts import (
    ResearchBrief,
    ResearchTodo,
    ScopedRequest,
    Stage,
)


def _activity(state, label: str, status: str = "success"):
    return [
        *state.get("activity", []),
        {"id": f"{len(state.get('activity', [])) + 1}:{label}", "label": label, "status": status},
    ]


def _scope_from_state(state) -> ScopedRequest:
    value = state.get("scoped_request")
    if value and all(
        key in value
        for key in (
            "objective",
            "research_boundaries",
            "constraints",
            "success_criteria",
            "deliverable_format",
        )
    ):
        return ScopedRequest.model_validate(value)
    brief = ResearchBrief.model_validate(value)
    return ScopedRequest(
        objective=brief.research_question,
        research_boundaries="",
        constraints="",
        success_criteria="",
        deliverable_format="technical report",
    )


class ResearchNodes:
    """Concrete, dependency-injected nodes for the post-Planning workflow."""

    def __init__(self, vfs, index, mcp):
        self.vfs = vfs
        self.index = index
        self.mcp = mcp

    def researching(self, state):
        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        if not todos:
            raise ValueError("Research requires a persisted non-empty todo plan")
        artifacts = list(state.get("artifacts", []))
        evidence = list(state.get("evidence", []))
        index_refs = list(state.get("index_refs", []))
        completed = []
        for todo in todos:
            todo, refs, chunks = run_worker(
                self.vfs,
                self.index,
                state["run_id"],
                todo,
                self.mcp,
            )
            completed.append(todo.model_dump(mode="json"))
            artifacts.extend(ref.model_dump(mode="json") for ref in refs)
            evidence.extend(
                ref.model_dump(mode="json")
                for ref in refs
                if ref.kind == "evidence_bundle"
            )
            index_refs.extend(chunk.chunk_id for chunk in chunks)
        return {
            **state,
            "stage": Stage.RESEARCHING.value,
            "current_action": "Research tasks completed",
            "activity": _activity(state, "Calling MCP research tools"),
            "todos": completed,
            "todo_ids": [todo["todo_id"] for todo in completed],
            "artifacts": artifacts,
            "evidence": evidence,
            "index_refs": index_refs,
        }

    def research_complete(self, state):
        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        assessment = assess(
            todos,
            state.get("artifacts", []),
            state.get("index_refs", []),
        )
        ref = self.vfs.write_json(
            state["run_id"],
            "supervisor/progress/latest.json",
            assessment.model_dump(mode="json"),
            "progress_assessment",
        )
        if not assessment.ready_for_synthesis:
            raise RuntimeError("ResearchComplete denied: " + assessment.next_action)
        return {
            **state,
            "stage": Stage.SYNTHESIZING.value,
            "current_action": "ResearchComplete validated",
            "activity": _activity(state, "Assessing research completeness"),
            "progress_assessment": assessment.model_dump(mode="json"),
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }

    def synthesizing(self, state):
        scope = _scope_from_state(state)
        report = synthesize(
            scope,
            vfs=self.vfs,
            run_id=state["run_id"],
            index=self.index,
        )
        ref = self.vfs.write(
            state["run_id"],
            "reports/latest.md",
            report,
            "latest_report",
        )
        return {
            **state,
            "stage": Stage.VERIFYING.value,
            "current_action": "Draft synthesized from indexed evidence",
            "activity": _activity(state, "Synthesizing report"),
            "previous_draft": state.get("draft", ""),
            "draft": report,
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
            "verification": None,
        }

    def verifying(self, state):
        scope = _scope_from_state(state)
        brief = ResearchBrief.model_validate(state["scoped_request"])
        chunks = self.index.retrieve(
            brief.research_question,
            limit=20,
            run_id=state["run_id"],
        )
        result = verify(state.get("draft", ""), scope, chunks)
        ref = self.vfs.write_json(
            state["run_id"],
            f"verification/result.{result.verification_id}.json",
            result.model_dump(mode="json"),
            "verification",
        )
        history = [*state.get("verification_history", []), result.model_dump(mode="json")]
        return {
            **state,
            "stage": (
                Stage.AWAITING_APPROVAL.value
                if result.passed
                else Stage.NEEDS_REVIEW.value
            ),
            "status": "paused" if result.passed else "needs_review",
            "current_action": "Awaiting approval" if result.passed else "Verification needs review",
            "activity": _activity(state, "Verifying report"),
            "verification": result.model_dump(mode="json"),
            "verification_history": history,
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }
