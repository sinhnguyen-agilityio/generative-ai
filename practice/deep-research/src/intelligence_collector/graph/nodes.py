"""Graph node implementations.

Every node is a ``state -> state update`` callable. Nodes that need
collaborators (LLM agents, VFS, index, MCP) are grouped into small classes so
dependencies are injected once and the wiring in ``build.py`` stays declarative.
"""

from __future__ import annotations

import hashlib

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
    FeedbackDecision,
    IssueType,
    ResearchBrief,
    ResearchTodo,
    ScopeAssessment,
    Stage,
    VerificationResult,
)
from intelligence_collector.graph.state import GraphState
from intelligence_collector.observability.evaluation import FaithfulnessEvaluator
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
    WRITE_BRIEF = "write_research_brief"
    PLAN = "planning"
    # Research pipeline
    RESEARCH = "researching"
    ASSESS_RESEARCH = "research_complete"
    # Build-Verify-Fix loop
    SYNTHESIZE = "synthesizing"
    VERIFY = "verifying"
    EVALUATE_FAITHFULNESS = "evaluating_faithfulness"
    FIX = "fixing"
    RESEARCH_GAPS = "researching_gaps"
    NEEDS_REVIEW = "needs_review"
    # Human review and publication
    WRITER = "writer"
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
        report_guidelines: list[str] = (),
        max_concurrent_researchers: int = DEFAULT_MAX_CONCURRENT_RESEARCHERS,
        recovery_budget: int = DEFAULT_RECOVERY_BUDGET,
        evidence_top_k: int = DEFAULT_EVIDENCE_TOP_K,
        faithfulness_evaluator: FaithfulnessEvaluator | None = None,
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
            self.supervisor,
            report_writer or ReportWriter(researcher.router),
            report_judge=report_judge,
            report_guidelines=report_guidelines,
            recovery_budget=recovery_budget,
            evidence_top_k=evidence_top_k,
            faithfulness_evaluator=faithfulness_evaluator,
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
            artifacts.extend(ref.model_dump(mode="json") for ref in refs)
            evidence.extend(
                ref.model_dump(mode="json")
                for ref in refs
                if ref.kind == "evidence_bundle"
            )
            index_refs.extend(chunk.chunk_id for chunk in chunks)
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
        todos = [ResearchTodo.model_validate(
            todo) for todo in state.get("todos", [])]
        assessment = self.supervisor.assess(
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
            raise RuntimeError(
                "ResearchComplete denied: "
                + ("; ".join(assessment.blockers) or assessment.next_action)
            )
        return {
            **state,
            "stage": Stage.SYNTHESIZING.value,
            "current_action": "ResearchComplete validated",
            "activity": _activity(state, "Assessing research completeness"),
            "progress_assessment": assessment.model_dump(mode="json"),
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }


def _merge_research(state, results):
    """Fold supervisor results into the accumulated todos/artifacts/evidence/index refs."""
    todos = list(state.get("todos", []))
    artifacts = list(state.get("artifacts", []))
    evidence = list(state.get("evidence", []))
    index_refs = list(state.get("index_refs", []))
    for todo, refs, chunks in results:
        todos.append(todo.model_dump(mode="json"))
        artifacts.extend(ref.model_dump(mode="json") for ref in refs)
        evidence.extend(ref.model_dump(mode="json")
                        for ref in refs if ref.kind == "evidence_bundle")
        index_refs.extend(chunk.chunk_id for chunk in chunks)
    return {"todos": todos, "artifacts": artifacts, "evidence": evidence, "index_refs": index_refs}


# ===== BUILD-VERIFY-FIX =====


class BVFNodes:
    """Mandatory Build-Verify-Fix loop around the report.

    Build (``synthesizing``) -> Verify (``verifying``, one LLM-as-judge call — see
    ``agents/report_judge.py`` — judging goal achievement, per-question coverage, fair
    representation of the evidence, uncertainty disclosure, plain language, and safe content) ->
    on failure Fix (``fixing``: record the attempt and reflect to learn lessons) -> either a
    revised draft (DRAFT_DEFECT) or targeted gap research (EVIDENCE_GAP, when the judge decides
    the evidence itself is insufficient) -> Build again. Only a report that passes every criterion
    leaves the loop normally; exhausting the recovery budget or repeating the same failures ends
    in ``needs_review``.
    """

    def __init__(
        self,
        vfs,
        index,
        supervisor: ResearchSupervisor,
        writer: ReportWriter,
        *,
        report_judge: ReportJudge,
        report_guidelines: list[str] = (),
        recovery_budget: int,
        evidence_top_k: int = DEFAULT_EVIDENCE_TOP_K,
        faithfulness_evaluator: FaithfulnessEvaluator | None = None,
    ):
        self.vfs = vfs
        self.index = index
        self.supervisor = supervisor
        self.writer = writer
        self.report_judge = report_judge
        self.report_guidelines = list(report_guidelines)
        self.recovery_budget = recovery_budget
        self.evidence_top_k = evidence_top_k
        self.faithfulness_evaluator = faithfulness_evaluator

    def _catalog(self, state):
        """Full, unfiltered catalog — used by Verify and gap research, never narrowed."""
        return load_catalog(self.vfs, state["run_id"], state.get("evidence", []))

    def synthesizing(self, state):
        """Build the first draft, or a Fix revision applying every automated lesson learned so
        far. A human reviewer's formatting request never comes here — see ``writing``."""
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
        report = self.writer.write(
            brief,
            catalog,
            revision=revision,
            guidelines=self.report_guidelines,
            run_id=state["run_id"],
            thread_id=state.get("thread_id", ""),
        )
        version = state.get("recovery_attempt", 0) + 1
        ref = persist_report(self.vfs, state["run_id"], report, version)
        return {
            **state,
            "stage": Stage.VERIFYING.value,
            "current_action": (
                f"Draft v{version} revised with {len(revision.lessons)} lesson(s)"
                if revision else "First draft built from stored evidence"
            ),
            "activity": _activity(state, "Synthesizing report"),
            "previous_draft": state.get("draft", ""),
            "draft": report,
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
            "verification": None,  # a changed artifact never inherits a previous verification
        }

    def writing(self, state):
        """The ``writer`` node: presentation step between the verified report and the human review (``approval``).

        Straight from verification (no request): the report is presented as-is and remembered
        as ``verified_draft``. With a reviewer's formatting request (routed here by
        ``classify_feedback``): the current report is re-presented in the requested format by
        ``ReportWriter.reformat`` — no re-synthesis and no re-verification; the reformat prompt
        forbids changing any claim, number or citation. Publication stays bound to the verified
        source (``verified_draft``) and to the exact text the reviewer approves.
        """
        draft = state.get("draft", "")
        verification = state.get("verification") or {}
        verified_draft = state.get("verified_draft", "")
        if draft and hashlib.sha256(draft.encode()).hexdigest() == verification.get("artifact_hash"):
            verified_draft = draft  # fresh from verification
        base = {**state, "verified_draft": verified_draft, "human_feedback": "",
                "stage": Stage.AWAITING_APPROVAL.value, "status": "paused"}

        request = state.get("human_feedback", "")
        if not request:
            return {**base, "format_unchanged": False, "current_action": "Report ready for review",
                    "activity": _activity(state, "Presenting report for review")}

        report = self.writer.reformat(
            draft, request, run_id=state["run_id"], thread_id=state.get("thread_id", ""),
        )
        revision = state.get("format_revision", 0) + 1
        ref = self.vfs.write(state["run_id"], f"reports/formatted.v{revision}.md", report, "draft")
        unchanged = report.strip() == draft.strip()
        return {
            **base,
            "current_action": (
                "The formatting request left the report unchanged" if unchanged
                else f"Report reformatted per the reviewer's request (format v{revision})"
            ),
            "format_unchanged": unchanged,
            "format_revision": revision,
            "activity": _activity(state, "Reformatting report"),
            "previous_draft": draft,
            "draft": report,
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }

    def verifying(self, state):
        """LLM-as-judge is the sole Verify gate: one reasoning-tier call (no search/scrape tools)
        re-reads the report against the gathered evidence and decides whether it is factually
        faithful and follows the configured report guidelines — see ``agents/report_judge.py``."""
        catalog = self._catalog(state)
        researched = list(state.get("researched_gaps", []))
        brief = ResearchBrief.model_validate(state["scoped_request"])
        criteria = build_report_judge_criteria(self.report_guidelines)
        report = state.get("draft", "")
        verdict = self.report_judge.verify(
            brief,
            criteria,
            report,
            format_evidence(catalog),
            catalog,
            researched_gaps=researched,
            run_id=state["run_id"],
            thread_id=state.get("thread_id", ""),
        )
        artifact_hash = hashlib.sha256(report.encode()).hexdigest() if report else ""
        result = judge_verdict_to_result(verdict, artifact_hash)
        fingerprint = failure_fingerprint(result)
        stalled = not result.passed and fingerprint == state.get(
            "progress_fingerprint")
        run_id = state["run_id"]
        save_catalog_state(self.vfs, run_id, state.get(
            "evidence", []), researched)
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
            "verification_history": [
                *state.get("verification_history", []
                           ), result.model_dump(mode="json")
            ],
            "progress_fingerprint": fingerprint,
            "stall_count": state.get("stall_count", 0) + 1 if stalled else 0,
            "recovery_budget": state.get("recovery_budget", self.recovery_budget),
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }

    def evaluating_faithfulness(self, state):
        """RAGAS Faithfulness over the final report only.

        Reached from ``verifying`` exactly once, via the graph edge added only on the PASSED
        route (see ``build.py``) — never as part of a build/verify/fix attempt, so a report
        that fails and gets rewritten is never scored along the way, only the one draft that
        actually left the BVF loop. Observability only, per research.md's "Observability and
        evaluation" decision: a judge-LLM failure here is recorded, not raised, and never
        re-enters the loop.
        """
        faithfulness_score = None
        faithfulness_error = ""
        refs = []
        if self.faithfulness_evaluator is not None:
            catalog = self._catalog(state)
            brief = ResearchBrief.model_validate(state["scoped_request"])
            draft = state.get("draft", "")
            contexts = cited_source_excerpts(draft, catalog)
            verification_id = (state.get("verification") or {}).get("verification_id", "")
            try:
                faithfulness_score = self.faithfulness_evaluator(
                    brief.research_question, draft, contexts
                )
            except Exception as exc:
                faithfulness_error = str(exc)
            refs.append(self.vfs.write_json(
                state["run_id"],
                f"verification/faithfulness.{verification_id}.json",
                {
                    "score": faithfulness_score,
                    "error": faithfulness_error,
                    "question": brief.research_question,
                    "context_count": len(contexts),
                },
                "faithfulness",
            ))
        return {
            **state,
            "artifacts": [*state.get("artifacts", []), *(r.model_dump(mode="json") for r in refs)],
            "faithfulness_score": faithfulness_score,
            "faithfulness_error": faithfulness_error,
        }

    def fixing(self, state):
        """Self-improving retry: record the failure, and for draft defects reflect before fixing."""
        verification = VerificationResult.model_validate(state["verification"])
        attempt = state.get("recovery_attempt", 0) + 1
        budget = state.get("recovery_budget", self.recovery_budget)
        history = list(state.get("bvf_history", []))
        lessons = list(state.get("bvf_lessons", []))
        entry = {
            "attempt": attempt,
            "issue_type": verification.issue_type.value if verification.issue_type else None,
            "failed_checks": [c.name for c in verification.check_results if not c.passed],
            "fingerprint": state.get("progress_fingerprint", ""),
            "stalled": state.get("stall_count", 0) > 0,
        }
        strategy = ""
        if verification.issue_type == IssueType.DRAFT_DEFECT:
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
            action = "revising the draft"
        else:
            entry["gaps"] = verification.affected_ids
            action = "researching evidence gaps"
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
            "stage": (
                Stage.RESEARCHING if verification.issue_type == IssueType.EVIDENCE_GAP
                else Stage.SYNTHESIZING
            ).value,
            "recovery_attempt": attempt,
            "current_action": f"BVF fix attempt {attempt}/{budget}: {action}",
            "activity": _activity(state, f"Fix attempt {attempt}: {action}"),
            "bvf_history": history,
            "bvf_lessons": lessons,
            "fix_strategy": strategy,
            "artifacts": [*state.get("artifacts", []), *(ref.model_dump(mode="json") for ref in refs)],
        }

    def researching_gaps(self, state):
        """EVIDENCE_GAP repair: one targeted research sub-agent per unresearched gap."""
        catalog = self._catalog(state)
        researched = list(state.get("researched_gaps", []))
        gaps = [gap for gap in catalog.gaps if gap.gap_id not in researched]
        todos = [
            ResearchTodo(
                research_question=f"{gap.text} (Context: {gap.question})")
            for gap in gaps
        ]
        results = self.supervisor.delegate(
            state["run_id"], todos, thread_id=state.get("thread_id", "")) if todos else []
        merged = {**state, **_merge_research(state, results)}
        # Each gap gets one targeted research round; whatever remains must be disclosed.
        researched += [gap.gap_id for gap in gaps]
        researched += [gap.gap_id for gap in self._catalog(
            merged).gaps if gap.gap_id not in researched]
        return {
            **merged,
            "stage": Stage.SYNTHESIZING.value,
            "current_action": f"Researched {len(gaps)} evidence gap(s)",
            "activity": _activity(state, "Researching evidence gaps"),
            "todo_ids": [todo["todo_id"] for todo in merged["todos"]],
            "researched_gaps": researched,
        }

    def needs_review(self, state):
        """Recovery budget exhausted or no progress: stop without ever labelling it verified."""
        verification = state.get("verification") or {}
        attempts = state.get("recovery_attempt", 0)
        reason = (
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
            "last_error": verification.get("repair_details", ""),
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
    """The human-in-the-loop review, reached through ``writer`` once a report has fully passed
    verification (and faithfulness scoring):

    * ``approval`` interrupts the graph for one decision on the report ``writer`` presented,
      binding it to that exact text: approve -> ``publishing``; reject -> ``classify_feedback``.
    * ``classify_feedback`` asks an LLM (``feedback_router``) what the reviewer asked for:
      - ``format`` (presentation only): back to ``writer``, which re-presents the same report in
        the requested format — no re-synthesis, no re-verification — then back to ``approval``;
      - ``scope`` (a change in what the report covers): back to ``clarify_with_user``, with the
        feedback as a new user turn, so the scope agent re-assesses it (it may ask a clarifying
        question first), rewrites the research brief, and the planner then plans new questions,
        re-entering the full Build-Verify-Fix loop.
      Keeping the LLM call in its own node means a failed classification is retried on its
      own, never re-prompting the reviewer for a decision already given.
    * ``publishing`` writes the approved text to the permanent output location and ends the run.
    """

    def __init__(self, vfs, output_dir: str, feedback_router=None):
        self.vfs = vfs
        self.output_dir = output_dir
        self.feedback_router = feedback_router

    def approval(self, state: GraphState) -> GraphState:
        from langgraph.types import interrupt

        from intelligence_collector.publication import approval_for

        decision = interrupt({"kind": "publication_approval", "run_id": state.get("run_id")})
        verification = VerificationResult.model_validate(state["verification"])
        # The reviewer decides on the text writer presented, so the approval binds to it.
        approval = approval_for(
            hashlib.sha256(state.get("draft", "").encode()).hexdigest(),
            verification.verification_id,
            bool(decision.get("approved")),
            str(decision.get("actor") or "local-user"),
        )
        if approval.approved:
            return {
                **state,
                "approval": approval.model_dump(mode="json"),
                "current_action": "Reviewer approved — publishing",
                "activity": _activity(state, "Reviewer approved the report"),
            }
        # Rejected: classify_feedback decides where the feedback is addressed.
        return {
            **state,
            "approval": approval.model_dump(mode="json"),
            "human_feedback": str(decision.get("feedback") or "").strip(),
            "feedback_route": "",
            "current_action": "Reviewer rejected — classifying feedback",
            "activity": _activity(state, "Reviewer rejected the report", "failure"),
        }

    def publishing(self, state: GraphState) -> GraphState:
        """Write the approved text, hash-bound to the approval and to its verified source."""
        from intelligence_collector.publication import publish

        run_id = state["run_id"]
        destination = f"{self.output_dir.rstrip('/')}/{run_id}.md"
        record = publish(
            state.get("draft", ""),
            destination,
            ApprovalDecision.model_validate(state["approval"]),
            VerificationResult.model_validate(state["verification"]),
            verified_source=state.get("verified_draft") or state.get("draft", ""),
        )
        ref = self.vfs.write_json(
            run_id, "publication/record.json", record.model_dump(mode="json"), "publication_record",
        )
        return {
            **state,
            "publication": record.model_dump(mode="json"),
            "stage": (Stage.PUBLISHED if record.published else Stage.AWAITING_APPROVAL).value,
            "current_action": (
                f"Published to {destination}" if record.published
                else f"Publication failed: {record.error}"
            ),
            "activity": _activity(
                state, "Publishing report", "success" if record.published else "failure"
            ),
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }

    def classify_feedback(self, state: GraphState) -> GraphState:
        """LLM step: route the reviewer's rejection feedback to a rewrite or a re-scope."""
        feedback = state.get("human_feedback", "")
        if self.feedback_router is not None and feedback:
            verdict = self.feedback_router.route(
                feedback, state.get("draft", ""),
                run_id=state.get("run_id", ""), thread_id=state.get("thread_id", ""),
            )
        else:
            # No classifier configured, or the reviewer rejected without feedback: default to
            # the cheaper format path rather than discarding the whole research plan.
            verdict = FeedbackDecision(route="format", reason="No feedback to classify; defaulting to format.")

        history = [*state.get("bvf_history", []), {
            "attempt": state.get("recovery_attempt", 0),
            "issue_type": "HUMAN_FEEDBACK",
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
        if verdict.route == "format":
            # writer consumes human_feedback and re-presents the report in the requested format.
            return {**state, **update, "human_feedback": feedback}
        # Scope change: the feedback becomes the next user turn of the scope conversation. The
        # new brief starts a fresh Build-Verify-Fix cycle, so the attempt/stall counters restart.
        return {
            **state, **update,
            "request": feedback,
            "human_feedback": "",
            "verified_draft": "",
            "format_revision": 0,
            "recovery_attempt": 0,
            "stall_count": 0,
            "stage": Stage.SCOPING.value,
        }


# ===== STAGE-ONLY STUBS =====


def stage_only(name: str):
    """Placeholder node that only records its own name as the stage."""

    def advance(state: GraphState) -> GraphState:
        return {**state, "stage": name}

    return advance
