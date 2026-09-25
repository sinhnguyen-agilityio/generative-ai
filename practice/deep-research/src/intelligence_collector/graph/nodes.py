"""Graph node implementations.

Every node is a ``state -> state update`` callable. Nodes that need
collaborators (LLM agents, VFS, index, MCP) are grouped into small classes so
dependencies are injected once and the wiring in ``build.py`` stays declarative.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from intelligence_collector.agents.supervisor import assess
from intelligence_collector.agents.synthesis import synthesize
from intelligence_collector.agents.verification import verify
from intelligence_collector.agents.worker import run_worker
from intelligence_collector.contracts import (
    ResearchBrief,
    ResearchTodo,
    ScopeAssessment,
    ScopedRequest,
    Stage,
)
from intelligence_collector.graph.state import GraphState


class Node:
    """Stable node identifiers shared by graph wiring, UI progress, tracing and tests."""

    # Scope and planning
    CLARIFY = "clarify_with_user"
    WRITE_BRIEF = "write_research_brief"
    PLAN = "planning"
    # Research pipeline
    RESEARCH = "researching"
    ASSESS_RESEARCH = "research_complete"
    SYNTHESIZE = "synthesizing"
    VERIFY = "verifying"
    # Publication
    APPROVE = "approval"
    # Placeholder used by the stage-only graph when no Scope Agent is injected
    STUB_SCOPE = "scope"


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


# ===== SCOPE AND PLANNING =====


class ScopeNodes:
    """Scope clarification and the research brief."""

    def __init__(self, scope_agent, vfs=None):
        self.scope_agent = scope_agent
        self.vfs = vfs

    def clarify_with_user(self, state: GraphState) -> GraphState:
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
            **state,
            "stage": Stage.SCOPING.value,
            "messages": [*history, HumanMessage(content=request), AIMessage(content=reply)],
            "scope_assessment": assessment.model_dump(mode="json"),
        }

    def write_research_brief(self, state: GraphState) -> GraphState:
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
            **state,
            "stage": Stage.PLANNING.value,
            "scoped_request": brief.model_dump(mode="json"),
        }


class PlannerNodes:
    """Planning: the Planner Agent turns the brief into questions and calls `write_todos`."""

    def __init__(self, planner_agent, vfs=None):
        self.planner_agent = planner_agent
        self.vfs = vfs

    def planning(self, state: GraphState) -> GraphState:
        if self.vfs is None:
            raise RuntimeError("Planning requires an injected VFS")
        brief = ResearchBrief.model_validate(state["scoped_request"])
        todos = self.planner_agent.write_todos(
            brief,
            self.vfs,
            state["run_id"],
            thread_id=state.get("thread_id", ""),
        )
        return {
            **state,
            "stage": Stage.PLANNING.value,
            "todos": [todo.model_dump(mode="json") for todo in todos],
        }


# ===== RESEARCH PIPELINE =====


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


# ===== APPROVAL =====


def approval(state: GraphState) -> GraphState:
    """Pause the graph until a human resumes it with a publication decision."""
    from langgraph.types import interrupt

    decision = interrupt({"kind": "publication_approval", "run_id": state.get("run_id")})
    stage = Stage.PUBLISHED if decision.get("approved") else Stage.AWAITING_APPROVAL
    return {**state, "approval": decision, "stage": stage.value}


# ===== STAGE-ONLY STUBS =====


def stage_only(name: str):
    """Placeholder node that only records its own name as the stage."""

    def advance(state: GraphState) -> GraphState:
        return {**state, "stage": name}

    return advance
