"""Research pipeline: delegation to sub-agents and the readiness assessment."""

from __future__ import annotations

import json

from intelligence_collector.ports import ArtifactStore, VectorIndex
from intelligence_collector.agents.report_judge import ReportJudge
from intelligence_collector.agents.report_writer import (
    DEFAULT_EVIDENCE_TOP_K,
    ReportWriter,
)
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.supervisor import (
    DEFAULT_MAX_CONCURRENT_RESEARCHERS,
    ResearchSupervisor,
)
from intelligence_collector.agents.worker import _evidence_path, research_prefix
from intelligence_collector.contracts import (
    ResearchReadiness,
    ResearchTodo,
    Stage,
    TodoStatus,
    VerificationIssue,
)
from intelligence_collector.graph.nodes.common import (
    DEFAULT_RECOVERY_BUDGET,
    _activity,
    invalidate_draft_authority,
)
from intelligence_collector.graph.nodes.report import BVFNodes
from intelligence_collector.observability.evaluation import (
    ContextPrecisionEvaluator,
    FaithfulnessEvaluator,
    ResponseRelevancyEvaluator,
)


class ResearchNodes:
    """Concrete, dependency-injected nodes for the post-Planning workflow."""

    def __init__(
        self,
        vfs: ArtifactStore,
        index: VectorIndex | None,
        researcher: ResearchAgent,
        *,
        report_writer: ReportWriter | None = None,
        report_judge: ReportJudge,
        report_guidelines: list[str] = (),
        max_concurrent_researchers: int = DEFAULT_MAX_CONCURRENT_RESEARCHERS,
        recovery_budget: int = DEFAULT_RECOVERY_BUDGET,
        evidence_top_k: int = DEFAULT_EVIDENCE_TOP_K,
        faithfulness_evaluator: FaithfulnessEvaluator | None = None,
        require_faithfulness: bool = False,
        context_precision_evaluator: ContextPrecisionEvaluator | None = None,
        response_relevancy_evaluator: ResponseRelevancyEvaluator | None = None,
    ):
        self.vfs = vfs
        self.index = index
        self.supervisor = ResearchSupervisor(
            researcher,
            vfs,
            index,
            max_concurrent_researchers=max_concurrent_researchers,
        )
        self.bvf = BVFNodes(
            vfs,
            index,
            report_writer or ReportWriter(researcher.router),
            report_judge=report_judge,
            report_guidelines=report_guidelines,
            recovery_budget=recovery_budget,
            evidence_top_k=evidence_top_k,
            faithfulness_evaluator=faithfulness_evaluator,
            require_faithfulness=require_faithfulness,
            context_precision_evaluator=context_precision_evaluator,
            response_relevancy_evaluator=response_relevancy_evaluator,
        )

    def researching(self, state):
        """The supervisor model delegates the planned questions to research sub-agents.

        When the run has a plan record, retrieval requires an approval of exactly that plan. The
        supervisor may propose one replacement plan per approved original plan.
        """
        plan = state.get("plan")
        if plan is not None:
            approval = state.get("plan_approval") or {}
            if not approval.get("approved") or approval.get("plan_hash") != plan.get("plan_hash"):
                raise ValueError("Research requires human approval of the current plan")
        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        if not todos:
            raise ValueError("Research requires a persisted non-empty todo plan")
        results = self.supervisor.delegate(
            state["run_id"],
            todos,
            thread_id=state.get("thread_id", ""),
            brief=(state.get("scoped_request") or {}).get("research_question", ""),
            allow_replan=plan is not None and plan.get("kind") != "replacement",
        )
        replan = getattr(results, "replan", None)
        artifacts = list(state.get("artifacts", []))
        evidence = list(state.get("evidence", []))
        index_refs = list(state.get("index_refs", []))
        completed = []
        for todo, refs, chunks in results:
            completed.append(todo.model_dump(mode="json"))
            paths = {item.get("path") for item in artifacts}
            artifacts.extend(ref.model_dump(mode="json") for ref in refs if ref.path not in paths)
            evidence_paths = {item.get("path") for item in evidence}
            evidence.extend(
                ref.model_dump(mode="json")
                for ref in refs
                if ref.kind == "evidence_bundle" and ref.path not in evidence_paths
            )
            index_refs.extend(
                chunk.chunk_id for chunk in chunks if chunk.chunk_id not in index_refs
            )
        done = sum(todo["status"] == TodoStatus.COMPLETED for todo in completed)
        return {
            "stage": Stage.RESEARCHING.value,
            "current_action": "Supervisor proposed a replacement plan"
            if replan
            else f"Research sub-agents finished {done}/{len(completed)} tasks",
            "research_replan": {"reason": replan.reason, "todo_ids": replan.todo_ids}
            if replan
            else None,
            "activity": _activity(state, "Delegating research to sub-agents"),
            "todos": completed,
            "todo_ids": [todo["todo_id"] for todo in completed],
            "artifacts": artifacts,
            "evidence": evidence,
            "index_refs": index_refs,
        }

    def research_complete(self, state):
        """Assess enough evidence for the goal; optional gaps never force research.

        A task that failed technically (provider error, crashed sub-agent) becomes a limitation
        when other tasks produced usable evidence. Only when nothing usable exists does it stop
        the run, and then it goes straight to review: re-reading the same stored failure cannot
        change the outcome, so it is never retried here.
        """
        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        if not todos:
            return {
                "verification_issue": VerificationIssue.RESEARCH_FAILURE.value,
                "progress_assessment": None,
                "last_error": "No research todos; repair the scope or plan",
            }
        gaps, limitations, usable, technical = [], [], [], []
        for todo in todos:
            if todo.status == TodoStatus.FAILED:
                try:
                    failure = json.loads(
                        self.vfs.read(state["run_id"], f"{research_prefix(todo)}/outcome.json")
                    )
                except FileNotFoundError:
                    failure = {}
                if failure.get("failure_kind") == "technical" or (
                    "failure_kind" not in failure
                    and failure.get("error") not in {None, "Research returned no sources"}
                ):
                    technical.append(todo.todo_id)
                    limitations.append(f"Research failed technically for: {todo.research_question}")
                    gaps.append(
                        {
                            "todo_id": todo.todo_id,
                            "instruction": "Find direct evidence answering: "
                            + todo.research_question,
                        }
                    )
                    continue
            try:
                bundle = json.loads(self.vfs.read(state["run_id"], _evidence_path(todo)))
            except FileNotFoundError:
                bundle = {}
            # Corrupt/unreadable storage is a technical exception, not a content gap.
            if bundle.get("sources") and bundle.get("summary"):
                usable.append(todo.todo_id)
                limitations.extend(bundle.get("unresolved_gaps") or [])
                if bundle.get("sufficient") is not True:
                    limitations.append(
                        f"Evidence completeness not established for: {todo.research_question}"
                    )
            else:
                limitations.append(f"No usable evidence for: {todo.research_question}")
                gaps.append(
                    {
                        "todo_id": todo.todo_id,
                        "instruction": "Find direct evidence answering: " + todo.research_question,
                    }
                )
        if technical and not usable:
            # research_gaps names the failed tasks so the recovery "research" action reopens them.
            return {
                "verification_issue": VerificationIssue.RESEARCH_FAILURE.value,
                "progress_assessment": None,
                "research_gaps": gaps,
                "last_error": "Research failed technically; retry or check provider configuration",
            }
        status = (
            (ResearchReadiness.PARTIAL if limitations else ResearchReadiness.READY)
            if usable
            else ResearchReadiness.INSUFFICIENT
        )
        exhausted = state.get("research_gap_attempt", 0) >= min(
            state.get("research_gap_budget", 1), 1
        )
        ready = status != ResearchReadiness.INSUFFICIENT or exhausted
        assessment = {
            "status": status.value,
            "ready_for_synthesis": ready,
            "blockers": limitations,
            "next_action": "synthesize" if ready else "gap research",
        }
        ref = self.vfs.write_json(
            state["run_id"], "supervisor/progress/latest.json", assessment, "progress_assessment"
        )
        return {
            "stage": Stage.SYNTHESIZING.value if ready else Stage.RESEARCHING.value,
            "current_action": f"Research {status.value}: synthesize best supported answer"
            if ready
            else "Research missing core evidence once",
            "activity": _activity(state, "Assessing research completeness"),
            "progress_assessment": assessment,
            "research_status": status.value,
            "research_limitations": limitations,
            # Known gaps go to the writer as limitations; whether the report still answers what
            # was asked is decided by the coverage gate, so a gap never skips report approval.
            "result_incomplete": False,
            "research_gaps": gaps,
            "verification_issue": (
                VerificationIssue.NONE if ready else VerificationIssue.EVIDENCE_GAP
            ).value,
            "research_gap_budget": min(state.get("research_gap_budget", 1), 1),
            "last_error": "\n".join(limitations) if not ready else "",
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }

    def prepare_gap_research(self, state):
        if state.get("research_gap_attempt", 0) >= min(state.get("research_gap_budget", 1), 1):
            raise ValueError("Research gap allowance exhausted")
        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        gaps = {gap["todo_id"]: gap["instruction"] for gap in state.get("research_gaps", [])}
        if not gaps or not set(gaps).issubset({todo.todo_id for todo in todos}):
            raise ValueError("Gap research requires known affected todo ids")
        for todo in todos:
            if todo.todo_id in gaps:
                todo.status = TodoStatus.PLANNED
                todo.research_attempt += 1
                todo.research_instruction = gaps[todo.todo_id]
        ref = self.vfs.write_json(
            state["run_id"],
            "planning/active_todos.json",
            [todo.model_dump(mode="json") for todo in todos],
            "todos",
        )
        return {
            # New evidence leads to a new draft; nothing earned by the current one carries over.
            **invalidate_draft_authority(),
            "todos": [todo.model_dump(mode="json") for todo in todos],
            "research_gap_attempt": state.get("research_gap_attempt", 0) + 1,
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }
