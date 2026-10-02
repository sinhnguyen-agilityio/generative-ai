"""Graph node implementations.

Every node is a ``state -> state update`` callable. Nodes that need
collaborators (LLM agents, VFS, index, MCP) are grouped into small classes so
dependencies are injected once and the wiring in ``build.py`` stays declarative.
"""

from __future__ import annotations

import hashlib
import json
from intelligence_collector.agents.worker import _evidence_path, research_prefix

from langchain_core.messages import AIMessage, HumanMessage

from intelligence_collector.agents.report_judge import (
    ReportJudge,
    build_report_judge_criteria,
    judge_verdict_to_result,
)
from intelligence_collector.agents.report_writer import (
    DEFAULT_EVIDENCE_TOP_K,
    ReportWriter,
    RevisionContext,
    format_evidence,
    merge_lessons,
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
    RecoveryAction,
    RecoveryDecision,
    ResearchBrief,
    ResearchReadiness,
    ResearchTodo,
    ScopeAssessment,
    Stage,
    VerificationResult,
    TodoStatus,
)
from intelligence_collector.graph.state import GraphState
from intelligence_collector.observability.evaluation import (
    ContextPrecisionEvaluator,
    FaithfulnessEvaluator,
    ResponseRelevancyEvaluator,
)
from intelligence_collector.source_verification import (
    cited_source_excerpts,
    failure_fingerprint,
    load_catalog,
    save_catalog_state,
)

DEFAULT_RECOVERY_BUDGET = 3
# Identical failures on this many consecutive re-verifications, despite reflection, ends the loop.
MAX_STALLED_ATTEMPTS = 2


class Node:
    """Stable node identifiers shared by graph wiring, UI progress, tracing and tests."""

    # Scope and planning
    CLARIFY = "clarify_with_user"
    AWAIT_CLARIFICATION = "await_clarification"
    WRITE_BRIEF = "write_research_brief"
    PLAN = "planning"
    # Research pipeline
    RESEARCH = "researching"
    ASSESS_RESEARCH = "research_complete"
    # Build-Verify-Fix loop
    SYNTHESIZE = "synthesizing"
    PREPARE_REVISION = "prepare_revision"
    VERIFY = "verifying"
    EVALUATE_QUALITY = "evaluating_quality"
    FIX = "fixing"
    NEEDS_REVIEW = "needs_review"
    AWAIT_RECOVERY = "await_recovery"
    APPLY_RECOVERY = "apply_recovery_decision"
    PREPARE_GAP_RESEARCH = "prepare_gap_research"
    FINISH_INCOMPLETE = "finish_incomplete"
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
        update = {**state, "recovery_decision": None, "recovery_route": action.value,
                  "bvf_history": history, "approval": None, "publication": None}
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
            "draft": "",
            "verification": None,
            "research_status": "UNKNOWN", "research_limitations": [], "result_incomplete": False,
            "research_gap_attempt": 0, "research_gap_budget": 1,
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
            context_precision_evaluator=context_precision_evaluator,
            response_relevancy_evaluator=response_relevancy_evaluator,
        )

    def researching(self, state):
        """Supervisor delegates each planned question to its own research sub-agent."""
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
        """Assess enough evidence for the goal; optional gaps never force research."""
        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        if not todos:
            return {**state, "verification_issue": "research_verifier_failure",
                    "research_verifier_attempt": state.get("research_verifier_attempt", 0) + 1,
                    "last_error": "No research todos; repair the scope or plan"}
        gaps, limitations, usable = [], [], []
        for todo in todos:
            if todo.status == TodoStatus.FAILED:
                try:
                    failure = json.loads(self.vfs.read(state["run_id"], f"{research_prefix(todo)}/outcome.json"))
                except FileNotFoundError:
                    failure = {}
                if failure.get("failure_kind") == "technical" or (
                    "failure_kind" not in failure and failure.get("error") not in {None, "Research returned no sources"}
                ):
                    return {**state, "verification_issue": "research_verifier_failure",
                            "progress_assessment": None,
                            "research_verifier_attempt": state.get("research_verifier_attempt", 0) + 1,
                            "last_error": "Research failed technically; retry or check provider configuration"}
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
        status = (ResearchReadiness.PARTIAL if limitations else ResearchReadiness.READY) if usable else ResearchReadiness.INSUFFICIENT
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
            "result_incomplete": status != ResearchReadiness.READY,
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
    """Mandatory Build-Verify-Fix loop around the report.

    Build (``synthesizing``) -> Verify (``verifying``, one LLM-as-judge call — see
    ``agents/report_judge.py`` — that says which criteria failed and why) -> on failure Fix
    (``fixing``: record the attempt and reflect to learn lessons) -> Build again, rewriting the
    report from the same evidence. No new research happens inside the loop: a claim the evidence
    cannot support is removed, narrowed, or disclosed under ``## Limitations``. Only a report that
    passes every criterion leaves the loop normally; exhausting the recovery budget or repeating
    the same failures ends in ``needs_review``.
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
        context_precision_evaluator: ContextPrecisionEvaluator | None = None,
        response_relevancy_evaluator: ResponseRelevancyEvaluator | None = None,
    ):
        self.vfs = vfs
        self.index = index
        self.writer = writer
        self.report_judge = report_judge
        self.report_guidelines = list(report_guidelines)
        self.recovery_budget = recovery_budget
        self.evidence_top_k = evidence_top_k
        self.faithfulness_evaluator = faithfulness_evaluator
        self.context_precision_evaluator = context_precision_evaluator
        self.response_relevancy_evaluator = response_relevancy_evaluator

    def _catalog(self, state):
        """Full, unfiltered catalog — used by Verify and quality scoring, never narrowed."""
        return load_catalog(self.vfs, state["run_id"], state.get("evidence", []),
                            active_todo_ids=[todo["todo_id"] for todo in state["todos"]]
                            if "todos" in state else None)

    def synthesizing(self, state):
        """Build the first draft, a Fix revision applying every lesson learned so far, or a
        rewrite of the reviewed report applying the human reviewer's requests.

        ``human_feedback`` is set only on the pass right after a reviewer's "revise" reply: the
        report they commented on is rewritten. ``reviewer_requests`` (every request since the
        last scope change) stays in force for this rewrite and any BVF fix that follows it.
        """
        brief = ResearchBrief.model_validate(state["scoped_request"])
        previous = state.get("verification")
        revision = None
        if previous and not previous.get("passed") and state.get("draft"):
            revision = RevisionContext(
                attempt=state.get("recovery_attempt", 0),
                failures=previous.get("repair_details", ""),
                previous_draft=state["draft"],
                lessons=list(state.get("bvf_lessons", [])),
                strategy=state.get("fix_strategy", ""),
            )
        # The writer's prompt sees a per-question top-K selection; Verify still checks the
        # full catalog independently, so narrowing here can never hide a citation problem.
        catalog = select_relevant_evidence(
            self._catalog(state), self.index, state["run_id"], top_k=self.evidence_top_k
        )
        reviewer_requests = list(state.get("reviewer_requests", []))
        reviewing = bool(state.get("human_feedback"))
        report = self.writer.write(
            brief,
            catalog,
            revision=revision,
            reviewer_requests=reviewer_requests,
            reviewed_report=state.get("draft", "") if reviewing else "",
            research_status=state.get("research_status", "READY"),
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
                f"Draft v{version} revised with {len(revision.lessons)} lesson(s)" if revision
                else f"Draft v{version} rewritten per the reviewer's request" if reviewing
                else "First draft built from stored evidence"
            ),
            "human_feedback": "",  # consumed
            "activity": _activity(state, "Synthesizing report"),
            "previous_draft": state.get("draft", ""),
            "draft": report,
            "draft_version": version,
            "verifier_attempt": 0,
            "verifier_error": "",
            "writer_evidence": writer_evidence,
            "artifacts": [
                *state.get("artifacts", []),
                ref.model_dump(mode="json"),
                ranking_ref.model_dump(mode="json"),
            ],
            "verification": None,  # a changed artifact never inherits a previous verification
        }

    def verifying(self, state):
        """Combine independent semantic judging with structural citation/request checks.

        Verification errors never count as either a passing or a failed report verdict.
        """
        catalog = self._catalog(state)
        brief = ResearchBrief.model_validate(state["scoped_request"])
        criteria = build_report_judge_criteria(self.report_guidelines)
        criteria.extend([
            "The report addresses the user's core goal within the available evidence, without "
            "substituting another topic; unavailable answers are explicitly identified, not invented.",
        ])
        if state.get("result_incomplete"):
            criteria.append("The report is explicitly labeled INCOMPLETE and accurately discloses "
                            "the unverified parts without implying that research is complete. Known limitations: " +
                            json.dumps(state.get("research_limitations", [])))
        report = state.get("draft", "")
        from intelligence_collector.agents.reviewer_checks import reviewer_format_checks

        requests = list(state.get("reviewer_requests", []))
        criteria.extend(
            f"The report satisfies this human reviewer request: {request}"
            for request in requests
        )
        try:
            verdict = self.report_judge.verify(
                brief,
                criteria,
                report,
                format_evidence(catalog),
                reviewer_requests=list(state.get("reviewer_requests", [])),
                run_id=state["run_id"],
                thread_id=state.get("thread_id", ""),
            )
        except Exception as exc:
            return {**state, "verification_issue": "verifier_failure", "verification": None,
                    "verifier_error": str(exc), "last_error": "Report verifier unavailable",
                    "verifier_attempt": state.get("verifier_attempt", 0) + 1}
        verdict.results.extend(reviewer_format_checks(report, requests))
        from intelligence_collector.source_verification import citation_errors
        from intelligence_collector.contracts import ReportJudgeCriterion

        for error in citation_errors(report, catalog):
            verdict.results.append(ReportJudgeCriterion(
                criterion="Citation integrity: " + error, met=False,
                justification=error, issue_type="draft_defect"))
        failed_items = [item for item in verdict.results if not item.met]
        issue = "draft_defect"
        missing = [item.justification for item in failed_items if item.issue_type == "evidence_gap"]
        incomplete = bool(state.get("result_incomplete")) or bool(missing)
        artifact_hash = hashlib.sha256(
            report.encode()).hexdigest() if report else ""
        result = judge_verdict_to_result(verdict, artifact_hash)
        fingerprint = failure_fingerprint(result)
        stalled = not result.passed and fingerprint == state.get(
            "progress_fingerprint")
        run_id = state["run_id"]
        save_catalog_state(self.vfs, run_id, state.get("evidence", []))
        ref = self.vfs.write_json(
            run_id,
            f"verification/result.{result.verification_id}.json",
            result.model_dump(mode="json"),
            "verification",
        )
        failed = [check.name for check in result.check_results if not check.passed]

        return {
            **state,
            "stage": (Stage.AWAITING_APPROVAL if result.passed else Stage.VERIFYING).value,
            "status": "paused" if result.passed else "active",
            "current_action": (
                f"All {len(result.checks)} judge criteria passed"
                if result.passed else f"Verification failed: {', '.join(failed)}"
            ),
            "activity": _activity(state, "Verifying report"),
            "verification": result.model_dump(mode="json"),
            "verification_issue": issue,
            "result_incomplete": incomplete,
            "research_status": "PARTIAL" if missing else state.get("research_status", "READY"),
            "research_limitations": list(dict.fromkeys([*state.get("research_limitations", []), *missing])),
            "verifier_attempt": 0,
            "verifier_error": "",
            "verification_history": [
                *state.get("verification_history", []
                           ), result.model_dump(mode="json")
            ],
            "progress_fingerprint": fingerprint,
            "stall_count": state.get("stall_count", 0) + 1 if stalled else 0,
            "recovery_budget": state.get("recovery_budget", self.recovery_budget),
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }

    def evaluating_quality(self, state):
        """RAGAS quality scores of the final report: Faithfulness, Context Precision and Response
        Relevancy.

        Reached from ``verifying`` exactly once, via the graph edge added only on the PASSED
        route (see ``build.py``) — never as part of a build/verify/fix attempt, so a report
        that fails and gets rewritten is never scored along the way, only the one draft that
        actually left the BVF loop. Observability only, per research.md's "Observability and
        evaluation" decision: each metric runs only when its evaluator is configured, and a
        judge-LLM failure is recorded for that metric alone, never raised, and never re-enters
        the loop.
        """
        update: dict = {
            "faithfulness_score": None, "faithfulness_error": "",
            "context_precision_score": None, "context_precision_error": "",
            "context_precision_details": [],
            "response_relevancy_score": None, "response_relevancy_error": "",
        }
        evaluators = (
            self.faithfulness_evaluator,
            self.context_precision_evaluator,
            self.response_relevancy_evaluator,
        )
        if state.get("result_incomplete") or all(evaluator is None for evaluator in evaluators):
            return {**state, **update}

        catalog = self._catalog(state)
        question = ResearchBrief.model_validate(
            state["scoped_request"]).research_question
        draft = state.get("draft", "")
        record: dict = {"question": question}

        def run(metric: str, evaluate):
            """Run one metric; a failure is recorded under ``<metric>_error``, never raised."""
            try:
                return evaluate()
            except Exception as exc:
                update[f"{metric}_error"] = str(exc)
                return None

        if self.faithfulness_evaluator is not None:
            contexts = cited_source_excerpts(draft, catalog)
            record["faithfulness_context_count"] = len(contexts)
            update["faithfulness_score"] = run(
                "faithfulness", lambda: self.faithfulness_evaluator(
                    question, draft, contexts),
            )
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

    def finish_incomplete(self, state):
        """Display a verified, explicitly incomplete answer; never publish it."""
        verification = state.get("verification") or {}
        digest = hashlib.sha256(state.get("draft", "").encode()).hexdigest()
        if not verification.get("passed") or verification.get("artifact_hash") != digest:
            raise ValueError("Incomplete answer must still pass exact-artifact verification")
        content = "# INCOMPLETE — not a published report\n\n" + state.get("draft", "")
        ref = self.vfs.write(state["run_id"], "reports/incomplete.md", content, "incomplete_report")
        return {**state, "stage": Stage.INCOMPLETE.value, "status": "incomplete",
                "result_incomplete": True, "approval": None, "publication": None,
                "current_action": "Incomplete answer: some information could not be verified",
                "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
                "messages": [*state.get("messages", []), AIMessage(content=content)]}

    def fixing(self, state):
        """Self-improving retry: record the failure and reflect on it (diagnosis, lessons, strategy)
        before the next rewrite."""
        verification = VerificationResult.model_validate(state["verification"])
        attempt = state.get("recovery_attempt", 0) + 1
        budget = state.get("recovery_budget", self.recovery_budget)
        history = list(state.get("bvf_history", []))
        lessons = list(state.get("bvf_lessons", []))
        entry = {
            "attempt": attempt,
            "failed_checks": [c.name for c in verification.check_results if not c.passed],
            "fingerprint": state.get("progress_fingerprint", ""),
            "stalled": state.get("stall_count", 0) > 0,
        }
        reflection = self.writer.reflect(
            state.get("draft", ""),
            verification,
            history=history,
            lessons=lessons,
            stalled=entry["stalled"],
            run_id=state["run_id"],
            thread_id=state.get("thread_id", ""),
        )
        lessons = merge_lessons(lessons, reflection.lessons)
        strategy = reflection.strategy
        entry.update(diagnosis=reflection.diagnosis,
                     lessons=reflection.lessons, strategy=strategy)
        history.append(entry)
        run_id = state["run_id"]
        refs = [
            self.vfs.write_json(
                run_id, "verification/bvf_history.json", history, "bvf_history"),
            self.vfs.write_json(
                run_id, "verification/lessons.json", lessons, "bvf_lessons"),
        ]
        return {
            **state,
            "stage": Stage.SYNTHESIZING.value,
            "recovery_attempt": attempt,
            "current_action": f"BVF fix attempt {attempt}/{budget}: revising the draft",
            "activity": _activity(state, f"Fix attempt {attempt}: revising the draft"),
            "bvf_history": history,
            "bvf_lessons": lessons,
            "fix_strategy": strategy,
            "artifacts": [*state.get("artifacts", []), *(ref.model_dump(mode="json") for ref in refs)],
        }

    def needs_review(self, state):
        """Recovery budget exhausted or no progress: stop without ever labelling it verified."""
        verification = state.get("verification") or {}
        attempts = state.get("recovery_attempt", 0)
        reason = state.get("last_error") if state.get("verification_issue") in {
            "verifier_failure", "research_verifier_failure", "publication_failure"} else (
            "research gap budget exhausted" if state.get("research_gap_attempt", 0) >=
            state.get("research_gap_budget", 2) else
            "no progress after reflection" if state.get("stall_count", 0) >= MAX_STALLED_ATTEMPTS
            else f"recovery budget {state.get('recovery_budget', self.recovery_budget)} exhausted"
        )
        ref = self.vfs.write_json(
            state["run_id"],
            "verification/needs_review.json",
            {
                "status": Stage.NEEDS_REVIEW.value,
                "reason": reason,
                "attempts": attempts,
                "unresolved": verification.get("repair_details", ""),
                "lessons": state.get("bvf_lessons", []),
                "history": state.get("bvf_history", []),
            },
            "needs_review",
        )
        return {
            **state,
            "stage": Stage.NEEDS_REVIEW.value,
            "status": "needs_review",
            "last_error": verification.get("repair_details") or state.get("last_error", ""),
            "current_action": f"Needs review: {reason} after {attempts} fix attempt(s)",
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
            VerificationResult.model_validate({**state["verification"], "passed": False})
            if state.get("result_incomplete") else VerificationResult.model_validate(state["verification"]),
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
        """LLM step: read the reviewer's chat reply as an approval, a revision request or a scope
        change."""
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
