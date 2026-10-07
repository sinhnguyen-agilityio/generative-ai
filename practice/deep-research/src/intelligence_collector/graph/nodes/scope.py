"""Scope clarification, the research brief, and planning."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from intelligence_collector.agents.planner import plan_record
from intelligence_collector.contracts import (
    ResearchBrief,
    ScopeAssessment,
    Stage,
)
from intelligence_collector.graph.nodes.common import (
    invalidate_draft_authority,
    reset_research_rounds,
)
from intelligence_collector.graph.state import GraphState
from intelligence_collector.ports import ArtifactStore


class ScopeNodes:
    """Scope clarification and the research brief."""

    def __init__(self, scope_agent, vfs: ArtifactStore | None = None):
        self.scope_agent = scope_agent
        self.vfs = vfs

    def clarify_with_user(self, state: GraphState) -> GraphState:
        """Scope turn: guardrail and clarify decision. Appends the user's message and the agent's
        reply to ``messages``.
        """
        request = state.get("request", "")
        history = list(state.get("messages", []))
        assessment = self.scope_agent.clarify_with_user(
            request,
            history,
            run_id=state.get("run_id", ""),
            thread_id=state.get("thread_id", ""),
        )
        reply = assessment.question if assessment.need_clarification else assessment.verification
        return {
            "stage": Stage.SCOPING.value,
            "messages": [*history, HumanMessage(content=request), AIMessage(content=reply)],
            "scope_assessment": assessment.model_dump(mode="json"),
        }

    def write_research_brief(self, state: GraphState) -> GraphState:
        """Write the research brief once the scope is clear and move to planning."""
        assessment = ScopeAssessment.model_validate(state["scope_assessment"])
        brief = self.scope_agent.write_research_brief(
            state.get("request", ""),
            assessment,
            list(state.get("messages", [])),
            run_id=state.get("run_id", ""),
            thread_id=state.get("thread_id", ""),
            vfs=self.vfs,
        )
        return {
            "stage": Stage.PLANNING.value,
            "scoped_request": brief.model_dump(mode="json"),
        }


class PlannerNodes:
    """Planning: the Planner Agent turns the brief into research questions and calls
    `write_todos`."""

    def __init__(self, planner_agent, vfs: ArtifactStore | None = None):
        self.planner_agent = planner_agent
        self.vfs = vfs

    def planning(self, state: GraphState) -> GraphState:
        """Plan the research questions for the current brief. Also reached after a reviewer's
        scope change (``ApprovalNodes.approval`` -> scope clarification -> new brief -> here): the
        resulting ``todos`` fully replace the previous list, but ``evidence``/``index_refs`` are
        deliberately left to accumulate rather than reset: the run-scoped ``UrlRegistry``
        (``agents/worker.py::persist_research``) already dedupes a URL the new research re-finds,
        pointing it at its first capture instead of re-indexing it, so discarding old refs here
        would make that reused, already-good evidence invisible to ``research_complete``'s
        readiness check. ``draft``/``verification`` are reset: those are genuinely stale once
        the questions change."""
        if self.vfs is None:
            raise RuntimeError("Planning requires an injected VFS")
        brief = ResearchBrief.model_validate(state["scoped_request"])
        todos, shape = self.planner_agent.plan(
            brief, self.vfs, state["run_id"], thread_id=state.get("thread_id", "")
        )
        plan = plan_record(brief, todos, shape)
        version = state.get("plan_version", 0) + 1
        ref = self.vfs.write_json(
            state["run_id"], f"planning/plan.v{version}.json", plan, "research_plan"
        )
        return {
            # The questions changed, so the old draft and everything vouching for it are stale.
            **invalidate_draft_authority(),
            # A new plan starts research again at round 1.
            **reset_research_rounds(),
            "stage": Stage.PLANNING.value,
            "current_action": f"Plan with {len(todos)} research task(s) ready for approval",
            "todos": [todo.model_dump(mode="json") for todo in todos],
            "plan": plan,
            "plan_version": version,
            "plan_approval": None,
            "draft": "",
            "research_status": "UNKNOWN",
            "research_limitations": [],
            "result_incomplete": False,
            "research_gap_attempt": 0,
            "research_gap_budget": 1,
            "repair": None,
            "repair_researched": [],
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }
