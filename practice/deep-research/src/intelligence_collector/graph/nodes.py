"""Graph node implementations.

Every node is a ``state -> state update`` callable. Nodes that need
collaborators (LLM agents, VFS, index, MCP) are grouped into small classes so
dependencies are injected once and the wiring in ``build.py`` stays declarative.
"""

from __future__ import annotations

import hashlib
import json
import math
from intelligence_collector.agents.worker import _evidence_path, research_prefix

from langchain_core.messages import AIMessage, HumanMessage

from intelligence_collector.agents.planner import plan_record
from intelligence_collector.agents.report_judge import (
    ReportJudge,
    build_coverage_criteria,
    coverage_checks,
    verification_result,
)
from intelligence_collector.agents.report_writer import (
    DEFAULT_EVIDENCE_TOP_K,
    DIRECT_READ_CHARS,
    ReportWriter,
    format_evidence,
    select_relevant_evidence,
)
from intelligence_collector.agents.supervisor import (
    DEFAULT_MAX_CONCURRENT_RESEARCHERS,
    ResearchSupervisor,
)
from intelligence_collector.agents.synthesis import persist_report
from intelligence_collector.contracts import (
    ApprovalDecision,
    ClarificationResumeDecision,
    FeedbackDecision,
    PlanApprovalDecision,
    RecoveryAction,
    RecoveryDecision,
    ResearchBrief,
    ResearchReadiness,
    ResearchTodo,
    ScopeAssessment,
    Stage,
    VerificationCheck,
    VerificationResult,
    TodoStatus,
)
from intelligence_collector.graph.state import GraphState
from intelligence_collector.observability.evaluation import (
    ContextPrecisionEvaluator,
    FaithfulnessEvaluator,
    ResponseRelevancyEvaluator,
)
from intelligence_collector.repair import diagnose, item_key, locate_with_index
from intelligence_collector.source_verification import (
    _normalize_url,
    citation_errors,
    cited_source_excerpts,
    failure_fingerprint,
    load_catalog,
    save_catalog_state,
)

DEFAULT_RECOVERY_BUDGET = 3
# RAGAS Faithfulness must be strictly greater than this for the third gate to pass.
FAITHFULNESS_THRESHOLD = 0.90
# Identical failures on this many consecutive re-verifications, despite reflection, ends the loop.
MAX_STALLED_ATTEMPTS = 2
# Stored passages the extension judge reads for a reviewer's "extend" request.
EXTENSION_PASSAGES = 8


class Node:
    """Stable node identifiers shared by graph wiring, UI progress, tracing and tests."""

    # Scope and planning
    CLARIFY = "clarify_with_user"
    AWAIT_CLARIFICATION = "await_clarification"
    WRITE_BRIEF = "write_research_brief"
    PLAN = "planning"
    AWAIT_PLAN = "await_plan_approval"
    # Research pipeline
    RESEARCH = "researching"
    ASSESS_RESEARCH = "research_complete"
    # Build-Verify-Fix loop
    SYNTHESIZE = "synthesizing"
    PREPARE_REVISION = "prepare_revision"
    LOCATE_EXTENSION = "locate_extension_evidence"
    VERIFY = "verifying"
    EVALUATE_QUALITY = "evaluating_quality"
    FIX = "fixing"
    PREPARE_REPAIR_RESEARCH = "prepare_repair_research"
    REPLAN = "replan"
    NEEDS_REVIEW = "needs_review"
    AWAIT_RECOVERY = "await_recovery"
    APPLY_RECOVERY = "apply_recovery_decision"
    PREPARE_GAP_RESEARCH = "prepare_gap_research"
    RETRY_VERIFIER = "retry_verifier"
    RETRY_RESEARCH_ASSESSMENT = "retry_research_assessment"
    # Human review and publication
    APPROVE = "approval"
    CLASSIFY_FEEDBACK = "classify_feedback"
    PUBLISH = "publishing"
    # Placeholder used by the stage-only graph when no Scope Agent is injected
    STUB_SCOPE = "scope"


def _activity(state, label: str, status: str = "success"):
    return [
        *state.get("activity", []),
        {"id": f"{len(state.get('activity', [])) + 1}:{label}",
         "label": label, "status": status},
    ]


def _unresolved_lines(verification: dict, limit: int = 3) -> list[str]:
    """Short, user-facing lines for the failed checks of a verification record."""
    lines = []
    for check in (verification or {}).get("check_results", []):
        if check.get("passed"):
            continue
        detail = (check.get("details") or [""])[0]
        label = {"coverage": "Coverage", "citation": "Citations",
                 "faithfulness": "Faithfulness"}.get(check.get("gate"), "Check")
        lines.append(f"{label}: {detail[:220]}")
    return lines[:limit]


def await_clarification(state: GraphState) -> GraphState:
    """Pause after scope analysis and feed the validated answer back into scoping.

    Keeping this interrupt in its own node prevents the Scope Agent call in
    ``clarify_with_user`` from being repeated when LangGraph re-enters the interrupted node.
    """
    from langgraph.types import interrupt

    assessment = ScopeAssessment.model_validate(state["scope_assessment"])
    decision = ClarificationResumeDecision.model_validate(interrupt({
        "kind": "clarification_required",
        "run_id": state.get("run_id"),
        "question": assessment.question,
    }))
    return {
        **state,
        "request": decision.reply,
        "reviewer": decision.actor,
        "current_action": "Clarification received — reassessing scope",
    }


def await_plan_approval(state: GraphState) -> GraphState:
    """Pause before any retrieval until a human approves the exact plan (initial or replacement).

    No model call, tool call or artifact write happens before ``interrupt()``; the approval is
    bound to the plan hash, so a changed plan always needs a new approval.
    """
    from langgraph.types import interrupt

    plan = state.get("plan") or {}
    decision = PlanApprovalDecision.model_validate(interrupt({
        "kind": "plan_approval",
        "run_id": state.get("run_id"),
        "plan": plan,
    }))
    outcome = decision.outcome
    record = {"plan_hash": plan.get("plan_hash", ""), "approved": outcome == "approve",
              "outcome": outcome, "actor": decision.actor, "kind": plan.get("kind", "initial")}
    update = {**state, "plan_approval": record, "reviewer": decision.actor}
    if outcome == "approve":
        return {**update, "stage": Stage.RESEARCHING.value, "status": "active",
                "current_action": "Plan approved — starting research"}
    if outcome == "change":
        # The requested change becomes the next scope turn: new brief, new plan, new approval.
        return {**update, "stage": Stage.SCOPING.value, "request": decision.reply,
                "current_action": "Plan change requested — rescoping"}
    if plan.get("kind") == "replacement":
        return {**update, "stage": Stage.NEEDS_REVIEW.value,
                "current_action": "Replacement plan declined",
                "last_error": "The replacement research plan was declined."}
    return {**update, "stage": "Cancelled", "status": "cancelled",
            "current_action": "Plan declined; run cancelled"}


def await_recovery(state: GraphState, *, can_rescope=True) -> GraphState:
    """Pause a failed BVF run for an explicit, validated human recovery decision.

    The next node applies the stored decision through an explicit conditional graph edge.
    """
    from langgraph.types import interrupt

    decision = RecoveryDecision.model_validate(interrupt({
        "kind": "recovery_required",
        "run_id": state.get("run_id"),
        "reason": state.get("current_action") or state.get("last_error", ""),
        "allowed_actions": [action.value for action in RecoveryAction
                            if can_rescope or action is not RecoveryAction.RESCOPE],
    }))
    return {
        **state,
        "recovery_decision": decision.model_dump(mode="json"),
        "current_action": f"Recovery action authorized: {decision.action.value}",
    }


class RecoveryNodes:
    """Apply human authorization before any recovery work runs."""

    def __init__(self, vfs, *, can_rescope=False):
        self.vfs = vfs
        self.can_rescope = can_rescope

    def apply(self, state):
        decision = RecoveryDecision.model_validate(
            state.get("recovery_decision"))
        action = decision.action
        if action is RecoveryAction.RESCOPE and not self.can_rescope:
            raise ValueError("Rescoping requires a configured Scope Agent")
        history = [*state.get("bvf_history", []), {
            "kind": "RECOVERY_DECISION", **decision.model_dump(mode="json"),
        }]
        request = f"Recovery action: {action.value.replace('_', ' ')}" + (
            f" — {decision.feedback}" if decision.feedback else "")
        update = {**state, "recovery_decision": None, "recovery_route": action.value,
                  "bvf_history": history, "approval": None, "publication": None,
                  # The reviewer's decision is part of the chat, like a review reply.
                  "messages": [*state.get("messages", []), HumanMessage(content=request)]}
        if action is RecoveryAction.CANCEL:
            return {**update, "stage": "Cancelled", "status": "cancelled",
                    "current_action": "Run cancelled by reviewer"}
        if action is RecoveryAction.EXPORT_INCOMPLETE:
            content = (
                "# INCOMPLETE — unverified research report\n\n"
                "This artifact has unresolved verification issues and is not published.\n\n"
                + state.get("last_error", "") + "\n\n" + state.get("draft", "")
            )
            ref = self.vfs.write(state["run_id"], "reports/incomplete.md", content,
                                 "incomplete_report")
            return {**update, "stage": "Exported Incomplete", "status": "exported_incomplete",
                    "current_action": "Incomplete report saved for review",
                    "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")]}
        # Each human authorization starts a fresh, small recovery allowance. Verification history
        # and artifacts remain available, but no stale verification/approval follows the new draft.
        update.update(verification=None, status="active", recovery_attempt=0,
                      recovery_budget=decision.additional_attempts or 1,
                      stall_count=0, progress_fingerprint="", last_error="",
                      faithfulness_score=None, context_precision_score=None,
                      response_relevancy_score=None)
        update.update(research_gap_attempt=0, research_gap_budget=1,
                      research_verifier_attempt=0, verifier_attempt=0)
        if action in {RecoveryAction.RETRY_REPORT, RecoveryAction.REVISE}:
            feedback = decision.feedback or "Repair the unresolved verification issues: " + (
                state.get("last_error") or (state.get("verification")
                                            or {}).get("repair_details", "")
            )
            return {**update, "stage": Stage.SYNTHESIZING.value,
                    "human_feedback": feedback,
                    "reviewer_requests": [*state.get("reviewer_requests", []), feedback]}
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
            return {**update, "stage": Stage.RESEARCHING.value, "human_feedback": "",
                    "todos": [todo.model_dump(mode="json") for todo in todos],
                    "todo_ids": [todo.todo_id for todo in todos]}
        return {**update, "stage": Stage.SCOPING.value, "request": decision.feedback,
                "draft": "", "human_feedback": "", "reviewer_requests": []}


# ===== SCOPE AND PLANNING =====


class ScopeNodes:
    """Scope clarification and the research brief."""

    def __init__(self, scope_agent, vfs=None):
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
            **state,
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
            **state,
            "stage": Stage.PLANNING.value,
            "scoped_request": brief.model_dump(mode="json"),
        }


class PlannerNodes:
    """Planning: the Planner Agent turns the brief into research questions and calls
    `write_todos`."""

    def __init__(self, planner_agent, vfs=None):
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
        if hasattr(self.planner_agent, "plan"):
            todos, shape = self.planner_agent.plan(
                brief, self.vfs, state["run_id"], thread_id=state.get("thread_id", ""))
        else:
            todos, shape = self.planner_agent.write_todos(
                brief, self.vfs, state["run_id"], thread_id=state.get("thread_id", "")), ""
        plan = plan_record(brief, todos, shape)
        version = state.get("plan_version", 0) + 1
        ref = self.vfs.write_json(state["run_id"], f"planning/plan.v{version}.json", plan,
                                  "research_plan")
        return {
            **state,
            "stage": Stage.PLANNING.value,
            "current_action": f"Plan with {len(todos)} research task(s) ready for approval",
            "todos": [todo.model_dump(mode="json") for todo in todos],
            "plan": plan, "plan_version": version, "plan_approval": None,
            "draft": "",
            "verification": None,
            "research_status": "UNKNOWN", "research_limitations": [], "result_incomplete": False,
            "research_gap_attempt": 0, "research_gap_budget": 1,
            "repair": None, "repair_researched": [],
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }


# ===== RESEARCH PIPELINE =====


class ResearchNodes:
    """Concrete, dependency-injected nodes for the post-Planning workflow."""

    def __init__(
        self,
        vfs,
        index,
        researcher,
        *,
        report_writer: ReportWriter | None = None,
        report_judge: ReportJudge,
        research_assessor=None,
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
        self.research_assessor = research_assessor
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
        """Supervisor delegates each planned question to its own research sub-agent.

        When the run has a plan record, retrieval requires an approval of exactly that plan.
        """
        plan = state.get("plan")
        if plan is not None:
            approval = state.get("plan_approval") or {}
            if not approval.get("approved") or approval.get("plan_hash") != plan.get("plan_hash"):
                raise ValueError("Research requires human approval of the current plan")
        todos = [ResearchTodo.model_validate(
            todo) for todo in state.get("todos", [])]
        if not todos:
            raise ValueError(
                "Research requires a persisted non-empty todo plan")
        results = self.supervisor.delegate(
            state["run_id"], todos, thread_id=state.get("thread_id", "")
        )
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
            index_refs.extend(chunk.chunk_id for chunk in chunks if chunk.chunk_id not in index_refs)
        done = sum(todo["status"] == "completed" for todo in completed)
        return {
            **state,
            "stage": Stage.RESEARCHING.value,
            "current_action": f"Research sub-agents finished {done}/{len(completed)} tasks",
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
            return {**state, "verification_issue": "research_failure",
                    "progress_assessment": None,
                    "last_error": "No research todos; repair the scope or plan"}
        gaps, limitations, usable, technical = [], [], [], []
        for todo in todos:
            if todo.status == TodoStatus.FAILED:
                try:
                    failure = json.loads(self.vfs.read(state["run_id"], f"{research_prefix(todo)}/outcome.json"))
                except FileNotFoundError:
                    failure = {}
                if failure.get("failure_kind") == "technical" or (
                    "failure_kind" not in failure and failure.get("error") not in {None, "Research returned no sources"}
                ):
                    technical.append(todo.todo_id)
                    limitations.append(f"Research failed technically for: {todo.research_question}")
                    gaps.append({"todo_id": todo.todo_id,
                                 "instruction": "Find direct evidence answering: " + todo.research_question})
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
                    limitations.append(f"Evidence completeness not established for: {todo.research_question}")
            else:
                limitations.append(f"No usable evidence for: {todo.research_question}")
                gaps.append({"todo_id": todo.todo_id,
                             "instruction": "Find direct evidence answering: " + todo.research_question})
        if technical and not usable:
            # research_gaps names the failed tasks so the recovery "research" action reopens them.
            return {**state, "verification_issue": "research_failure",
                    "progress_assessment": None,
                    "research_gaps": gaps,
                    "last_error": "Research failed technically; retry or check provider configuration"}
        status =(ResearchReadiness.PARTIAL if limitations else ResearchReadiness.READY) if usable else ResearchReadiness.INSUFFICIENT
        if self.research_assessor is not None:
            try:
                verdict = self.research_assessor.assess(
                    (state.get("scoped_request") or {}).get("research_question", ""), todos,
                    format_evidence(load_catalog(self.vfs, state["run_id"], state.get("evidence", []),
                                                 active_todo_ids=[todo.todo_id for todo in todos])),
                    run_id=state["run_id"], thread_id=state.get("thread_id", ""))
                ids = [check.todo_id for check in verdict.questions]
                if len(ids) != len(todos) or set(ids) != {todo.todo_id for todo in todos}:
                    raise ValueError("Invalid assessment todo ids")
            except Exception:
                return {**state, "verification_issue": "research_verifier_failure",
                        "progress_assessment": None,
                        "research_verifier_attempt": state.get("research_verifier_attempt", 0) + 1,
                        "last_error": "Research evidence judge unavailable or returned invalid output"}
            status = verdict.status
            limitations = verdict.limitations
            gaps = [{"todo_id": check.todo_id, "instruction": check.missing_evidence}
                    for check in verdict.questions if not check.sufficient and check.missing_evidence.strip()]
            # A judge cannot grant readiness without any persisted evidence.
            if not usable:
                status = ResearchReadiness.INSUFFICIENT
                limitations = limitations or ["No usable evidence was retrieved"]
                gaps = gaps or [{"todo_id": todo.todo_id, "instruction": "Find direct evidence answering: " + todo.research_question}
                                for todo in todos]
        exhausted = state.get("research_gap_attempt", 0) >= min(state.get("research_gap_budget", 1), 1)
        ready = status != ResearchReadiness.INSUFFICIENT or exhausted
        assessment = {"status": status.value, "ready_for_synthesis": ready,
                      "blockers": limitations, "next_action": "synthesize" if ready else "gap research"}
        ref = self.vfs.write_json(state["run_id"], "supervisor/progress/latest.json",
                                  assessment, "progress_assessment")
        return {
            **state,
            "stage": Stage.SYNTHESIZING.value if ready else Stage.RESEARCHING.value,
            "current_action": f"Research {status.value}: synthesize best supported answer" if ready else "Research missing core evidence once",
            "activity": _activity(state, "Assessing research completeness"),
            "progress_assessment": assessment,
            "research_status": status.value,
            "research_limitations": limitations,
            # Known gaps go to the writer as limitations; whether the report still answers what
            # was asked is decided by the coverage gate, so a gap never skips report approval.
            "result_incomplete": False,
            "research_gaps": gaps,
            "verification_issue": "evidence_gap" if not ready else "",
            "research_verifier_attempt": 0,
            "research_gap_budget": min(state.get("research_gap_budget", 1), 1),
            "last_error": "\n".join(limitations) if not ready else "",
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }

    def prepare_gap_research(self, state):
        if state.get("research_gap_attempt", 0) >= min(state.get("research_gap_budget", 1), 1):
            raise ValueError("Research gap allowance exhausted")
        todos = [ResearchTodo.model_validate(
            todo) for todo in state.get("todos", [])]
        gaps = {gap["todo_id"]: gap["instruction"]
                for gap in state.get("research_gaps", [])}
        if not gaps or not set(gaps).issubset({todo.todo_id for todo in todos}):
            raise ValueError("Gap research requires known affected todo ids")
        for todo in todos:
            if todo.todo_id in gaps:
                todo.status = TodoStatus.PLANNED
                todo.research_attempt += 1
                todo.research_instruction = gaps[todo.todo_id]
        ref = self.vfs.write_json(state["run_id"], "planning/active_todos.json",
                                  [todo.model_dump(mode="json") for todo in todos], "todos")
        return {**state, "todos": [todo.model_dump(mode="json") for todo in todos],
                "research_gap_attempt": state.get("research_gap_attempt", 0) + 1,
                "verification": None, "approval": None, "publication": None,
                "verifier_attempt": 0,
                "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")]}


# ===== BUILD-VERIFY-FIX =====


class BVFNodes:
    """Build -> Verify -> (diagnose and repair) loop around the report.

    Build (``synthesizing``) writes a complete draft. Verify (``verifying``) runs exactly three
    gates on that exact draft: coverage of the required answers and output constraints (LLM
    judge), citation resolution against sources collected in this run, and RAGAS Faithfulness
    strictly greater than 0.90. PASS leaves the loop at once. FAIL enters ``fixing``, which
    diagnoses the smallest useful correction (see ``repair.py``) and routes to a targeted
    rewrite, Chroma-located evidence, targeted research or a replacement plan. A technical
    evaluator failure is ERROR: it is retried or escalated, never rewritten. At most
    ``recovery_budget`` improvement cycles run; exhaustion or no progress ends in
    ``needs_review``.
    """

    def __init__(
        self,
        vfs,
        index,
        writer: ReportWriter,
        *,
        report_judge: ReportJudge,
        report_guidelines: list[str] = (),
        recovery_budget: int,
        evidence_top_k: int = DEFAULT_EVIDENCE_TOP_K,
        faithfulness_evaluator: FaithfulnessEvaluator | None = None,
        require_faithfulness: bool = False,
        context_precision_evaluator: ContextPrecisionEvaluator | None = None,
        response_relevancy_evaluator: ResponseRelevancyEvaluator | None = None,
        planner=None,
    ):
        self.vfs = vfs
        self.index = index
        self.writer = writer
        self.report_judge = report_judge
        self.report_guidelines = list(report_guidelines)
        self.recovery_budget = recovery_budget
        self.evidence_top_k = evidence_top_k
        self.direct_read_chars = DIRECT_READ_CHARS
        self.faithfulness_evaluator = faithfulness_evaluator
        # When required, a missing evaluator is a configuration ERROR rather than a skipped gate.
        self.require_faithfulness = require_faithfulness
        self.context_precision_evaluator = context_precision_evaluator
        self.response_relevancy_evaluator = response_relevancy_evaluator
        # Injected by build_graph when a Planner exists; enables replacement plans.
        self.planner = planner
        # Injected by build_graph; without it an "extend" request always researches.
        self.extension_judge = None

    def _catalog(self, state):
        """Full, unfiltered catalog — used by Verify and quality scoring, never narrowed."""
        return load_catalog(self.vfs, state["run_id"], state.get("evidence", []),
                            active_todo_ids=[todo["todo_id"] for todo in state["todos"]]
                            if "todos" in state else None)

    def _writer_catalog(self, state, full, repair):
        """What the writer reads: small corpora directly, a large one narrowed via Chroma; a
        content repair reads the full stored evidence, and the Chroma route adds passages
        located for the affected items."""
        route = (repair or {}).get("route")
        if route in {"vfs", "research"}:
            return full
        narrowed = select_relevant_evidence(full, self.index, state["run_id"],
                                            top_k=self.evidence_top_k,
                                            direct_read_chars=self.direct_read_chars)
        if route == "chroma" and self.index is not None:
            return locate_with_index(self.vfs, self.index, state["run_id"], full, narrowed,
                                     repair["issues"], state.get("artifacts", []))
        return narrowed

    def synthesizing(self, state):
        """Build the first draft, a targeted repair of a draft that failed verification, or a
        rewrite of the reviewed report applying the human reviewer's requests.

        Every new version clears the previous verification, report approval and publication.
        """
        brief = ResearchBrief.model_validate(state["scoped_request"])
        repair = state.get("repair")
        catalog = self._writer_catalog(state, self._catalog(state), repair)
        reviewer_requests = list(state.get("reviewer_requests", []))
        reviewing = bool(state.get("human_feedback"))
        report = self.writer.write(
            brief,
            catalog,
            repair=repair,
            previous_draft=state.get("draft", ""),
            draft_version=state.get("draft_version", 0),
            report_shape=(state.get("plan") or {}).get("report_shape", ""),
            reviewer_requests=reviewer_requests,
            reviewed_report=state.get("draft", "") if reviewing else "",
            research_limitations=state.get("research_limitations", []),
            guidelines=self.report_guidelines,
            run_id=state["run_id"],
            thread_id=state.get("thread_id", ""),
        )
        version = state.get("draft_version", 0) + 1
        ref = persist_report(self.vfs, state["run_id"], report, version)
        # The ranked evidence this draft was written from, per research question: what Context
        # Precision scores if this draft becomes the final report.
        writer_evidence = [
            {"todo_id": bundle.todo_id, "question": bundle.question,
                "source_ids": bundle.source_ids}
            for bundle in catalog.bundles
        ]
        ranking_ref = self.vfs.write_json(
            state["run_id"], f"synthesis/retrieval.v{version}.json", writer_evidence, "retrieval",
        )
        return {
            **state,
            "stage": Stage.VERIFYING.value,
            "current_action": (
                f"Draft v{version} repaired ({repair['route']})" if repair
                else f"Draft v{version} rewritten per the reviewer's request" if reviewing
                else "First draft built from stored evidence"
            ),
            "human_feedback": "",  # consumed
            "activity": _activity(state, "Synthesizing report"),
            "previous_draft": state.get("draft", ""),
            "draft": report,
            "draft_version": version,
            "repair": None,  # consumed
            "verifier_attempt": 0,
            "verifier_error": "",
            "writer_evidence": writer_evidence,
            "artifacts": [
                *state.get("artifacts", []),
                ref.model_dump(mode="json"),
                ranking_ref.model_dump(mode="json"),
            ],
            # A changed artifact never inherits a previous verification, approval or publication.
            "verification": None, "verification_outcome": "", "coverage_cache": None,
            "approval": None, "publication": None, "faithfulness_score": None,
        }

    def _error(self, state, message: str, partial: dict | None = None):
        """ERROR: the evaluation could not produce a trustworthy verdict. Nothing is rewritten."""
        return {**state, "verification_issue": "verifier_failure", "verification": None,
                "verification_outcome": "ERROR", "verifier_error": message,
                "last_error": "Report verifier unavailable: " + message,
                "coverage_cache": partial,
                "verifier_attempt": state.get("verifier_attempt", 0) + 1}

    def _faithfulness(self, brief, report, catalog):
        """Third gate. Returns ``(check, score, error)``; ``error`` means ERROR."""
        if self.faithfulness_evaluator is None:
            return None, None, ("Faithfulness evaluator is not configured"
                                if self.require_faithfulness else None)
        contexts = cited_source_excerpts(report, catalog)
        try:
            result = self.faithfulness_evaluator(
                brief.original_question or brief.research_question, report, contexts)
        except Exception as exc:
            return None, None, f"Faithfulness evaluation failed: {type(exc).__name__}"
        score = result.get("score") if isinstance(result, dict) else result
        unsupported = list(result.get("unsupported") or []) if isinstance(result, dict) else []
        if (not isinstance(score, (int, float)) or isinstance(score, bool)
                or math.isnan(score) or not 0 <= score <= 1):
            return None, None, "Faithfulness returned no valid score"
        passed = score > FAITHFULNESS_THRESHOLD
        check = VerificationCheck(
            name="faithfulness", passed=passed, gate="faithfulness",
            details=[] if passed else [
                f"RAGAS Faithfulness {score:.2f} is not above {FAITHFULNESS_THRESHOLD:.2f}."
                + (" Unsupported statements: " + " | ".join(unsupported[:8]) if unsupported else "")],
            missing_points=[] if passed else unsupported[:8],
            affected_ids=[] if passed else ["faithfulness"],
        )
        return check, float(score), None

    def verifying(self, state):
        """Run the three gates on the exact current draft and record PASS, FAIL or ERROR.

        Coverage and citation run first; Faithfulness is scored only on a draft that passed
        both (a draft failing them is repaired anyway). An evaluator failure is ERROR and never
        becomes a FAIL or a PASS; a retry reuses saved coverage checks for the same draft.
        """
        catalog = self._catalog(state)
        brief = ResearchBrief.model_validate(state["scoped_request"])
        report = state.get("draft", "")
        artifact_hash = hashlib.sha256(report.encode()).hexdigest() if report else ""
        from intelligence_collector.agents.reviewer_checks import reviewer_format_checks

        requests = list(state.get("reviewer_requests", []))
        cache = state.get("coverage_cache") or {}
        if cache.get("artifact_hash") == artifact_hash and cache.get("checks"):
            checks = [VerificationCheck.model_validate(item) for item in cache["checks"]]
        else:
            criteria = build_coverage_criteria(brief, self.report_guidelines, requests)
            try:
                pairs = self.report_judge.assess(
                    brief, criteria, report, format_evidence(catalog),
                    reviewer_requests=requests, run_id=state["run_id"],
                    thread_id=state.get("thread_id", ""))
            except Exception as exc:
                return self._error(state, str(exc))
            checks = coverage_checks(pairs)
            for item in reviewer_format_checks(report, requests):
                checks.append(VerificationCheck(
                    name="reviewer_word_limit", passed=item.met, gate="coverage",
                    issue_type="presentation", item=item.criterion,
                    details=[] if item.met else [f"{item.criterion} — {item.justification}"]))
            for error in citation_errors(report, catalog):
                checks.append(VerificationCheck(
                    name="citation_resolution", passed=False, gate="citation",
                    issue_type="draft_defect" if error.startswith("Unknown") else "presentation",
                    details=[error], affected_ids=[f"citation:{error}"]))
        score = None
        if all(check.passed for check in checks):
            check, score, error = self._faithfulness(brief, report, catalog)
            if error:
                return self._error(state, error, {"artifact_hash": artifact_hash,
                                                  "checks": [c.model_dump(mode="json") for c in checks]})
            if check is not None:
                checks.append(check)
        result = verification_result(checks, artifact_hash)
        fingerprint = failure_fingerprint(result)
        stalled = not result.passed and fingerprint == state.get("progress_fingerprint")
        run_id = state["run_id"]
        save_catalog_state(self.vfs, run_id, state.get("evidence", []))
        ref = self.vfs.write_json(
            run_id,
            f"verification/result.{result.verification_id}.json",
            {**result.model_dump(mode="json"), "outcome": "PASS" if result.passed else "FAIL",
             "draft_version": state.get("draft_version", 0), "faithfulness_score": score},
            "verification",
        )
        failed = [check.name for check in result.check_results if not check.passed]
        return {
            **state,
            "stage": (Stage.AWAITING_APPROVAL if result.passed else Stage.VERIFYING).value,
            "status": "paused" if result.passed else "active",
            "current_action": (
                f"All {len(result.checks)} verification checks passed"
                if result.passed else f"Verification failed: {', '.join(failed)}"
            ),
            "activity": _activity(state, "Verifying report"),
            "verification": result.model_dump(mode="json"),
            "verification_outcome": "PASS" if result.passed else "FAIL",
            "verification_issue": "draft_defect",
            "faithfulness_score": score,
            "coverage_cache": None,
            "verifier_attempt": 0,
            "verifier_error": "",
            "last_error": "",
            "verification_history": [
                *state.get("verification_history", []), result.model_dump(mode="json")
            ],
            "progress_fingerprint": fingerprint,
            "stall_count": state.get("stall_count", 0) + 1 if stalled else 0,
            "recovery_budget": state.get("recovery_budget", self.recovery_budget),
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }

    def evaluating_quality(self, state):
        """Optional RAGAS observations of the passing report: Context Precision and Response
        Relevancy. Faithfulness was already scored as a verification gate and is not repeated.

        Each metric runs only when configured; a judge failure is recorded for that metric
        alone and never re-enters the loop.
        """
        update: dict = {
            "context_precision_score": None, "context_precision_error": "",
            "context_precision_details": [],
            "response_relevancy_score": None, "response_relevancy_error": "",
        }
        # Tell the user a (new) verified report is waiting, so a revision visibly completes.
        update["messages"] = [*state.get("messages", []), AIMessage(content=(
            f"Report draft v{state.get('draft_version', 1)} passed verification (coverage, "
            "citations, faithfulness) and is ready for your review below."))]
        if self.context_precision_evaluator is None and self.response_relevancy_evaluator is None:
            return {**state, **update}

        catalog = self._catalog(state)
        question = ResearchBrief.model_validate(
            state["scoped_request"]).research_question
        draft = state.get("draft", "")
        record: dict = {"question": question, "faithfulness_score": state.get("faithfulness_score")}

        def run(metric: str, evaluate):
            """Run one metric; a failure is recorded under ``<metric>_error``, never raised."""
            try:
                return evaluate()
            except Exception as exc:
                update[f"{metric}_error"] = str(exc)
                return None

        if self.context_precision_evaluator is not None:
            ranked = [
                {
                    "question": item["question"],
                    "contexts": [
                        catalog.sources[sid].excerpt
                        for sid in item["source_ids"]
                        if sid in catalog.sources and catalog.sources[sid].excerpt
                    ],
                }
                for item in state.get("writer_evidence", [])
            ]
            result = run(
                "context_precision", lambda: self.context_precision_evaluator(
                    ranked, draft),
            )
            if result is not None:
                update["context_precision_score"] = result["score"]
                update["context_precision_details"] = result["per_question"]
        if self.response_relevancy_evaluator is not None:
            update["response_relevancy_score"] = run(
                "response_relevancy", lambda: self.response_relevancy_evaluator(
                    question, draft),
            )

        verification_id = (state.get("verification") or {}
                           ).get("verification_id", "")
        ref = self.vfs.write_json(
            state["run_id"],
            f"verification/quality.{verification_id}.json",
            {**record, **update},
            "quality",
        )
        return {
            **state,
            **update,
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }

    def fixing(self, state):
        """Open an improvement cycle: diagnose the smallest useful correction for the failed
        checks (no model call) and route to it. The repair plan is durable in state and VFS."""
        verification = VerificationResult.model_validate(state["verification"])
        attempt = state.get("recovery_attempt", 0) + 1
        budget = state.get("recovery_budget", self.recovery_budget)
        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        shown = {item["todo_id"]: item["source_ids"] for item in state.get("writer_evidence", [])}
        repair = diagnose(
            verification, todos=todos, full_catalog=self._catalog(state),
            shown_source_ids=shown, researched_items=list(state.get("repair_researched", [])),
            can_replan=self.planner is not None,
        )
        repair["cycle"] = attempt
        history = [*state.get("bvf_history", []), {
            "attempt": attempt, "draft_version": state.get("draft_version", 0),
            "route": repair["route"],
            "failed_checks": [c.name for c in verification.check_results if not c.passed],
            "issues": repair["issues"], "fingerprint": state.get("progress_fingerprint", ""),
        }]
        refs = [
            self.vfs.write_json(state["run_id"], f"verification/repair.{attempt}.json", repair,
                                "repair_plan"),
            self.vfs.write_json(state["run_id"], "verification/bvf_history.json", history,
                                "bvf_history"),
        ]
        return {
            **state,
            "stage": Stage.SYNTHESIZING.value,
            "recovery_attempt": attempt,
            "repair": repair,
            "current_action": f"Improvement cycle {attempt}/{budget}: {repair['route']} repair",
            "activity": _activity(state, f"Cycle {attempt}: {repair['route']} repair"),
            "bvf_history": history,
            "artifacts": [*state.get("artifacts", []), *(ref.model_dump(mode="json") for ref in refs)],
        }

    def prepare_repair_research(self, state):
        """Reopen only the affected approved tasks for targeted Tavily research on the missing
        points; completed evidence stays and other tasks are not researched again."""
        repair = state.get("repair") or {}
        targets: dict[str, list[str]] = {}
        for issue in repair.get("issues", []):
            if issue["route"] == "research" and issue["todo_id"]:
                targets.setdefault(issue["todo_id"], []).extend(
                    issue["missing_points"] or [issue["item"]])
        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        if not targets or not set(targets) <= {todo.todo_id for todo in todos}:
            raise ValueError("Targeted research requires known affected task ids")
        for todo in todos:
            if todo.todo_id in targets:
                todo.status = TodoStatus.PLANNED
                todo.research_attempt += 1
                todo.research_instruction = "Find evidence for: " + "; ".join(targets[todo.todo_id])
        researched = [*state.get("repair_researched", []), *(
            item_key(issue["item"]) for issue in repair.get("issues", [])
            if issue["route"] == "research")]
        return {**state, "todos": [todo.model_dump(mode="json") for todo in todos],
                "repair_researched": list(dict.fromkeys(researched)),
                # Targeted research replaces, not adds to, the pre-draft gap allowance.
                "research_gap_attempt": max(state.get("research_gap_attempt", 0), 1),
                "stage": Stage.RESEARCHING.value,
                "current_action": f"Targeted research for {len(targets)} task(s)"}

    def replan(self, state):
        """Targeted research did not help: propose replacement tasks for the unanswered items,
        keep the original checklist and completed evidence, and wait for plan approval."""
        if self.planner is None:
            raise ValueError("Replacement planning requires a Planner")
        brief = ResearchBrief.model_validate(state["scoped_request"])
        repair = state.get("repair") or {}
        unanswered = list(dict.fromkeys(
            issue["item"] for issue in repair.get("issues", []) if issue["route"] == "replan"))
        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        plan = self.planner.define_plan(
            brief, unanswered=unanswered,
            previous_questions=[todo.research_question for todo in todos],
            run_id=state["run_id"], thread_id=state.get("thread_id", ""))
        from intelligence_collector.agents.planner import create_todos

        new = create_todos(self.vfs, state["run_id"], plan.questions, plan.evidence_needed)
        record = plan_record(brief, [*todos, *new], plan.report_shape, kind="replacement")
        version = state.get("plan_version", 0) + 1
        ref = self.vfs.write_json(state["run_id"], f"planning/plan.v{version}.json", record,
                                  "research_plan")
        return {**state, "todos": [todo.model_dump(mode="json") for todo in [*todos, *new]],
                "plan": record, "plan_version": version, "plan_approval": None,
                "repair": {**repair, "route": "research"},
                "stage": Stage.PLANNING.value,
                "current_action": "Replacement plan awaiting approval",
                "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")]}

    def _extension_passages(self, state, catalog, topic: str) -> list:
        """Sources of this run's active evidence that the vector index ranks closest to `topic`,
        across all research tasks, at most ``EXTENSION_PASSAGES``. Hits that do not map to a
        source of the active catalog (an old scope, another run) are ignored."""
        if self.index is None:
            return []
        try:
            chunks = self.index.retrieve(topic, limit=EXTENSION_PASSAGES, run_id=state["run_id"])
        except Exception:
            return []
        by_url = {_normalize_url(source.url): sid for sid, source in catalog.sources.items()}
        found = dict.fromkeys(
            sid for chunk in chunks
            if (sid := by_url.get(_normalize_url(chunk.metadata.get("locator", "")))))
        return [catalog.sources[sid] for sid in found]

    def locate_extension_evidence(self, state):
        """Reviewer asked to add a section ("extend"): look for its evidence in what this run
        already stored before researching anything.

        The vector index finds the closest stored passages; the extension judge (one LLM call)
        decides whether they explain the topic. Enough -> a new completed task whose evidence
        bundle holds only the relevant passages, and the writer adds the section. Not enough,
        no passages, no judge, or a judge failure -> a new planned task researching only the
        missing points. Either way the task is added to the plan as an extension that the
        reviewer's request approves, and the existing report, brief and tasks are kept.
        """
        run_id = state["run_id"]
        brief = ResearchBrief.model_validate(state["scoped_request"])
        extension = dict(state.get("extension") or {})
        request = extension.get("request") or state.get("human_feedback", "")
        topic = (extension.get("topic") or request).strip()
        if not topic:
            raise ValueError("An extension request needs a topic")
        catalog = self._catalog(state)
        passages = self._extension_passages(state, catalog, topic)
        verdict, judge_error = None, ""
        if passages and self.extension_judge is not None:
            try:
                verdict = self.extension_judge.judge(
                    brief.original_question or brief.research_question, request, topic, passages,
                    run_id=run_id, thread_id=state.get("thread_id", ""))
            except Exception as exc:
                judge_error = f"{type(exc).__name__}: extension judge failed"
        offered = {source.source_id for source in passages}
        relevant = [sid for sid in dict.fromkeys(verdict.relevant_source_ids if verdict else [])
                    if sid in offered]
        # The judge's "sufficient" counts only with at least one real passage behind it.
        sufficient = bool(verdict and verdict.sufficient and relevant)
        missing = [point.strip() for point in (verdict.missing_points if verdict else [])
                   if point.strip()]

        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        evidence = list(state.get("evidence", []))
        artifacts = list(state.get("artifacts", []))
        if sufficient:
            todo = ResearchTodo(research_question=topic, evidence_needed=request,
                                status=TodoStatus.COMPLETED)
            bundle_ref = self.vfs.write_json(run_id, _evidence_path(todo), {
                "task_id": todo.todo_id,
                "research_attempt": todo.research_attempt,
                "question": topic,
                "summary": verdict.summary.strip() or f"Stored evidence on: {topic}",
                "sources": [{"locator": catalog.sources[sid].url,
                             "provenance": {"title": catalog.sources[sid].title},
                             "excerpt": catalog.sources[sid].excerpt} for sid in relevant],
                "sufficient": True,
                "unresolved_gaps": [],
                "indexed_chunks": [],
                "located_from_index": True,
            }, "evidence_bundle", task_id=todo.todo_id)
            evidence.append(bundle_ref.model_dump(mode="json"))
            artifacts.append(bundle_ref.model_dump(mode="json"))
        else:
            todo = ResearchTodo(research_question=topic, evidence_needed=request,
                                research_instruction="; ".join(missing))
        todos.append(todo)

        update: dict = {}
        if state.get("plan") is not None:
            # The reviewer's request is the approval of this one added task.
            plan = plan_record(brief, todos, state["plan"].get("report_shape", ""),
                               kind="extension")
            version = state.get("plan_version", 0) + 1
            plan_ref = self.vfs.write_json(run_id, f"planning/plan.v{version}.json", plan,
                                           "research_plan")
            artifacts.append(plan_ref.model_dump(mode="json"))
            update = {"plan": plan, "plan_version": version, "plan_approval": {
                "plan_hash": plan["plan_hash"], "approved": True, "outcome": "approve",
                "actor": state.get("reviewer") or "local-user", "kind": "extension"}}

        record = {
            "topic": topic, "request": request, "todo_id": todo.todo_id,
            "sufficient": sufficient,
            "passages": sorted(offered, key=lambda sid: int(sid[1:])),
            "relevant_source_ids": relevant, "missing_points": missing,
            "judge_error": judge_error,
        }
        count = sum(1 for item in state.get("bvf_history", []) if item.get("kind") == "EXTENSION")
        artifacts.append(self.vfs.write_json(
            run_id, f"extension/{count + 1}.json", record, "extension").model_dump(mode="json"))
        action = (f"Stored evidence covers the new section ({len(relevant)} source(s))"
                  if sufficient else f"Researching the new section: {topic}")
        return {
            **state, **update,
            "todos": [item.model_dump(mode="json") for item in todos],
            "todo_ids": [item.todo_id for item in todos],
            "evidence": evidence,
            "artifacts": artifacts,
            "extension": record,
            "bvf_history": [*state.get("bvf_history", []), {"kind": "EXTENSION", **record}],
            "stage": (Stage.SYNTHESIZING if sufficient else Stage.RESEARCHING).value,
            "current_action": action,
            "activity": _activity(state, "Locating evidence for the requested section"),
        }

    def needs_review(self, state):
        """Recovery budget exhausted or no progress: stop without ever labelling it verified."""
        verification = state.get("verification") or {}
        attempts = state.get("recovery_attempt", 0)
        reason = state.get("last_error") if state.get("verification_issue") in {
            "verifier_failure", "research_verifier_failure", "research_failure",
            "publication_failure"} else (
            "replacement plan declined" if (state.get("plan_approval") or {}).get("outcome") == "decline"
            else "research gap budget exhausted" if state.get("research_gap_attempt", 0) >=
            state.get("research_gap_budget", 2) and not state.get("draft") else
            "no progress after repair" if state.get("stall_count", 0) >= MAX_STALLED_ATTEMPTS
            else f"improvement cycle budget {state.get('recovery_budget', self.recovery_budget)} exhausted"
        )
        ref = self.vfs.write_json(
            state["run_id"],
            "verification/needs_review.json",
            {
                "status": Stage.NEEDS_REVIEW.value,
                "reason": reason,
                "attempts": attempts,
                "unresolved": verification.get("repair_details", ""),
                "history": state.get("bvf_history", []),
            },
            "needs_review",
        )
        version = state.get("draft_version", 0)
        notice = (f"Report draft v{version} could not be verified ({reason}). It is shown below "
                  "as an unverified draft; choose a recovery option to continue."
                  if state.get("draft") else f"No report could be verified ({reason}). Choose a "
                  "recovery option to continue.")
        notice += "".join(f"\n- {line}" for line in _unresolved_lines(verification))
        return {
            **state,
            "stage": Stage.NEEDS_REVIEW.value,
            "status": "needs_review",
            "messages": [*state.get("messages", []), AIMessage(content=notice)],
            "last_error": verification.get("repair_details") or state.get("last_error", ""),
            "current_action": f"Needs review: {reason} after {attempts} improvement cycle(s)",
            "activity": _activity(state, "Verification needs review", "failure"),
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }


# ===== APPROVAL =====


def approval(state: GraphState) -> GraphState:
    """Pause the graph until a human resumes it with a publication decision."""
    from langgraph.types import interrupt

    decision = interrupt({"kind": "publication_approval",
                         "run_id": state.get("run_id")})
    stage = Stage.PUBLISHED if decision.get(
        "approved") else Stage.AWAITING_APPROVAL
    return {**state, "approval": decision, "stage": stage.value}


class ApprovalNodes:
    """The human-in-the-loop review, reached once a report has fully passed verification (and
    faithfulness scoring):

    * ``approval`` interrupts the graph until the reviewer answers the verified report. The normal
      answer is a free-text chat reply (``{"reply": ...}``), always handed to
      ``classify_feedback``. An explicit ``{"approved": True}`` goes straight to ``publishing``;
      an explicit ``{"approved": False, "feedback": ...}`` is classified but can never be read as
      an approval.
    * ``classify_feedback`` asks an LLM (``feedback_router``) what the reply means:
      - ``approve``: the reply accepts the report as-is -> ``publishing``;
      - ``revise``: change the report using the same evidence -> ``synthesizing`` rewrites it
        with the reviewer's request, and it goes through Verify again before coming back here;
      - ``extend``: add a section within the same question -> ``locate_extension_evidence``
        uses stored evidence when it is enough and researches only what is missing, then the
        report is rewritten with the section and verified again;
      - ``scope``: the change needs different evidence -> ``clarify_with_user``, with the
        feedback as a new user turn, so the scope agent re-assesses it (it may ask a clarifying
        question first), rewrites the research brief, and the planner and researchers run again.
      Keeping the LLM call in its own node means a failed classification is retried on its
      own, never re-prompting the reviewer for a decision already given.
    * ``publishing`` writes the approved, verified report to the permanent output location.

    Approvals always bind to the hash of the exact verified report shown. The reviewer's replies
    and the agent's acknowledgements are appended to ``messages`` so the review reads as part of
    the chat conversation.
    """

    def __init__(self, vfs, output_dir: str, feedback_router=None):
        self.vfs = vfs
        self.output_dir = output_dir
        self.feedback_router = feedback_router

    @staticmethod
    def _approval_for(state: GraphState, approved: bool) -> dict:
        """The reviewer decides on the verified report shown, so the approval binds to it."""
        from intelligence_collector.publication import approval_for

        verification = VerificationResult.model_validate(state["verification"])
        return approval_for(
            hashlib.sha256(state.get("draft", "").encode()).hexdigest(),
            verification.verification_id,
            approved,
            state.get("reviewer") or "local-user",
        ).model_dump(mode="json")

    def approval(self, state: GraphState) -> GraphState:
        """Human review: pause (``interrupt``) for the reviewer's reply. An explicit approval goes
        to ``publishing``; a chat reply or a rejection goes to ``classify_feedback``.
        """
        from langgraph.types import interrupt

        decision = interrupt(
            {"kind": "publication_approval", "run_id": state.get("run_id")})
        from intelligence_collector.contracts import ApprovalResumeDecision

        decision = ApprovalResumeDecision.model_validate(decision).model_dump(mode="json")
        state = {
            **state, "reviewer": str(decision.get("actor") or "local-user")}
        if decision.get("approved") is True:
            return {
                **state,
                "approval": self._approval_for(state, True),
                "current_action": "Reviewer approved — publishing",
                "activity": _activity(state, "Reviewer approved the report"),
            }
        explicit_reject = decision.get("approved") is False
        return {
            **state,
            # A chat reply is undecided until classify_feedback reads it.
            "approval": self._approval_for(state, False) if explicit_reject else None,
            "human_feedback": str(decision.get("reply") or decision.get("feedback") or "").strip(),
            "feedback_route": "",
            "current_action": "Reading the reviewer's reply",
            "activity": _activity(state, "Reviewer replied to the report"),
        }

    def publishing(self, state: GraphState) -> GraphState:
        """Write the approved report, hash-bound to the approval and the passing verification."""
        from intelligence_collector.publication import publish

        run_id = state["run_id"]
        destination = f"{self.output_dir.rstrip('/')}/{run_id}.md"
        record = publish(
            state.get("draft", ""),
            destination,
            ApprovalDecision.model_validate(state["approval"]),
            VerificationResult.model_validate(state["verification"]),
        )
        ref = self.vfs.write_json(
            run_id, "publication/record.json", record.model_dump(
                mode="json"), "publication_record",
        )
        outcome = (
            f"Published to {destination}" if record.published
            else f"Publication failed: {record.error}"
        )
        return {
            **state,
            "publication": record.model_dump(mode="json"),
            "stage": (Stage.PUBLISHED if record.published else Stage.NEEDS_REVIEW).value,
            "approval": state.get("approval") if record.published else None,
            "verification": state.get("verification") if record.published else None,
            "last_error": "" if record.published else outcome,
            "verification_issue": "" if record.published else "publication_failure",
            "current_action": outcome,
            "messages": [*state.get("messages", []), AIMessage(content=outcome)],
            "activity": _activity(
                state, "Publishing report", "success" if record.published else "failure"
            ),
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }

    def classify_feedback(self, state: GraphState) -> GraphState:
        """LLM step: read the reviewer's chat reply as an approval, a revision request, an
        added section or a scope change."""
        feedback = state.get("human_feedback", "")
        if self.feedback_router is not None and feedback:
            verdict = self.feedback_router.route(
                feedback, state.get("draft", ""),
                run_id=state.get("run_id", ""), thread_id=state.get("thread_id", ""),
            )
        else:
            # No classifier configured, or no reply text: default to a revision rather than
            # publishing unasked or discarding the whole research plan.
            verdict = FeedbackDecision(
                route="revise", reason="No reply to classify; defaulting to revise.")
        explicit_reject = (state.get("approval") or {}
                           ).get("approved") is False
        explicit_accept = feedback.strip().casefold().rstrip(".! ") in {
            "approve", "approved", "i approve", "i agree", "đồng ý", "phê duyệt",
            "looks good, publish it"}
        if verdict.route == "approve" and (explicit_reject or not explicit_accept):
            verdict = FeedbackDecision(
                route="revise", reason="Publication requires an explicit, unconditional approval.")

        history = [*state.get("bvf_history", []), {
            "attempt": state.get("recovery_attempt", 0),
            "kind": "HUMAN_FEEDBACK",
            "route": verdict.route,
            "feedback": feedback,
            "reason": verdict.reason,
        }]
        update = {
            "feedback_route": verdict.route,
            "current_action": f"Reviewer feedback routed to {verdict.route}: {verdict.reason or feedback[:120]}",
            "activity": _activity(state, f"Reviewer feedback routed to {verdict.route}"),
            "bvf_history": history,
            "recovery_budget": state.get("recovery_budget", DEFAULT_RECOVERY_BUDGET),
        }
        # clarify_with_user records the reply itself on the scope path.
        replied = [*state.get("messages", []), HumanMessage(content=feedback)] if feedback else list(
            state.get("messages", []))
        if verdict.route == "approve":
            return {**state, **update, "messages": replied, "human_feedback": "",
                    "approval": self._approval_for(state, True)}
        if verdict.route == "revise":
            # synthesizing consumes human_feedback and rewrites the reviewed report; the request
            # joins reviewer_requests so later BVF fixes and the judge keep honouring it. A human
            # revision starts a fresh Build-Verify-Fix cycle, so the attempt/stall counters restart.
            requests = [*state.get("reviewer_requests", []), feedback] if feedback else list(
                state.get("reviewer_requests", []))
            return {
                **state, **update,
                "messages": [*replied, AIMessage(
                    content="Revising the report with your feedback; it will be verified again.")],
                "human_feedback": feedback or "(no specific request)",
                "reviewer_requests": requests,
                "verification": None,
                "approval": None,
                "publication": None,
                "recovery_attempt": 0,
                "stall_count": 0,
                "progress_fingerprint": "",
                "stage": Stage.SYNTHESIZING.value,
            }
        if verdict.route == "extend":
            # Keep the report, the brief and the plan; locate_extension_evidence looks for the
            # new section's evidence in the run's index and researches only what is missing.
            # The request joins reviewer_requests, so the coverage gate checks the new section.
            requests = [*state.get("reviewer_requests", []), feedback]
            return {
                **state, **update,
                "messages": [*replied, AIMessage(content=(
                    "Adding that to the report: checking the evidence already collected first, "
                    "then researching only what is missing. The updated report will be "
                    "verified again."))],
                "human_feedback": feedback,
                "reviewer_requests": requests,
                "extension": {"topic": (verdict.topic or feedback).strip(), "request": feedback},
                "verification": None,
                "approval": None,
                "publication": None,
                "recovery_attempt": 0,
                "stall_count": 0,
                "progress_fingerprint": "",
                "stage": Stage.RESEARCHING.value,
            }
        # Scope change: the feedback becomes the next user turn of the scope conversation. The
        # new brief starts a fresh research + Build-Verify-Fix cycle, so earlier reviewer requests
        # and the attempt/stall counters restart.
        return {
            **state, **update,
            "request": feedback,
            "human_feedback": "",
            "reviewer_requests": [],
            "verification": None,
            "approval": None,
            "publication": None,
            "recovery_attempt": 0,
            "stall_count": 0,
            "progress_fingerprint": "",
            "stage": Stage.SCOPING.value,
        }


# ===== STAGE-ONLY STUBS =====
def stage_only(name: str):
    """Placeholder node that only records its own name as the stage."""

    def advance(state: GraphState) -> GraphState:
        return {**state, "stage": name}

    return advance
