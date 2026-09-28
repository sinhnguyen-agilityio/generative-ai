"""Graph node implementations.

Every node is a ``state -> state update`` callable. Nodes that need
collaborators (LLM agents, VFS, index, MCP) are grouped into small classes so
dependencies are injected once and the wiring in ``build.py`` stays declarative.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from intelligence_collector.agents.acceptance_verifier import AcceptanceVerifier, merge_acceptance_verdict
from intelligence_collector.agents.report_writer import (
    DEFAULT_EVIDENCE_TOP_K,
    ReportWriter,
    RevisionContext,
    merge_lessons,
    select_relevant_evidence,
)
from intelligence_collector.agents.supervisor import (
    DEFAULT_MAX_CONCURRENT_RESEARCHERS,
    ResearchSupervisor,
)
from intelligence_collector.agents.synthesis import persist_report
from intelligence_collector.contracts import (
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
    verify_report,
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
    FIX = "fixing"
    RESEARCH_GAPS = "researching_gaps"
    NEEDS_REVIEW = "needs_review"
    # Publication
    APPROVE = "approval"
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
    """Planning: the Planner Agent turns the brief into questions, acceptance criteria, and
    calls `write_todos`."""

    def __init__(self, planner_agent, vfs=None):
        self.planner_agent = planner_agent
        self.vfs = vfs

    def planning(self, state: GraphState) -> GraphState:
        if self.vfs is None:
            raise RuntimeError("Planning requires an injected VFS")
        brief = ResearchBrief.model_validate(state["scoped_request"])
        todos, acceptance_criteria = self.planner_agent.write_todos(
            brief,
            self.vfs,
            state["run_id"],
            thread_id=state.get("thread_id", ""),
        )
        return {
            **state,
            "stage": Stage.PLANNING.value,
            "todos": [todo.model_dump(mode="json") for todo in todos],
            "acceptance_criteria": acceptance_criteria,
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
        acceptance_verifier: AcceptanceVerifier | None = None,
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
            acceptance_verifier=acceptance_verifier,
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

    Build (``synthesizing``) -> Verify (``verifying``, programmatic source-verification script,
    then, only if that passes, the LLM acceptance-criteria check) -> on failure Fix (``fixing``:
    record the attempt and reflect to learn lessons) -> either a revised draft (DRAFT_DEFECT) or
    targeted gap research (EVIDENCE_GAP) -> Build again. Only a report that passes every check
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
        acceptance_verifier: AcceptanceVerifier | None = None,
        recovery_budget: int,
        evidence_top_k: int = DEFAULT_EVIDENCE_TOP_K,
        faithfulness_evaluator: FaithfulnessEvaluator | None = None,
    ):
        self.vfs = vfs
        self.index = index
        self.supervisor = supervisor
        self.writer = writer
        self.acceptance_verifier = acceptance_verifier
        self.recovery_budget = recovery_budget
        self.evidence_top_k = evidence_top_k
        self.faithfulness_evaluator = faithfulness_evaluator

    def _catalog(self, state):
        """Full, unfiltered catalog — used by Verify and gap research, never narrowed."""
        return load_catalog(self.vfs, state["run_id"], state.get("evidence", []))

    def synthesizing(self, state):
        """Build the first draft, or a Fix revision that applies every lesson learned so far."""
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

    def verifying(self, state):
        """Run the programmatic source-verification script, then, only if it passes, the LLM
        acceptance-criteria check (there is no point judging goal achievement on a structurally
        broken draft)."""
        catalog = self._catalog(state)
        researched = list(state.get("researched_gaps", []))
        result = verify_report(state.get("draft", ""),
                               catalog, researched_gaps=researched)
        criteria = list(state.get("acceptance_criteria", []))
        brief = ResearchBrief.model_validate(state["scoped_request"])
        if result.passed and self.acceptance_verifier is not None and criteria:
            verdict = self.acceptance_verifier.verify(
                brief,
                criteria,
                state.get("draft", ""),
                run_id=state["run_id"],
                thread_id=state.get("thread_id", ""),
            )
            result = merge_acceptance_verdict(result, verdict)
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

        # RAGAS Faithfulness over the whole (now-passing) report — observability only, per
        # research.md's "Observability and evaluation" decision it never gates verification
        # or publication, so a judge-LLM failure here is recorded, not raised.
        faithfulness_score = None
        faithfulness_error = ""
        faithfulness_refs = []
        if result.passed and self.faithfulness_evaluator is not None:
            contexts = cited_source_excerpts(state.get("draft", ""), catalog)
            try:
                faithfulness_score = self.faithfulness_evaluator(
                    brief.research_question, state.get("draft", ""), contexts
                )
            except Exception as exc:
                faithfulness_error = str(exc)
            faithfulness_refs.append(self.vfs.write_json(
                run_id,
                f"verification/faithfulness.{result.verification_id}.json",
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
            "stage": (Stage.AWAITING_APPROVAL if result.passed else Stage.VERIFYING).value,
            "status": "paused" if result.passed else "active",
            "current_action": (
                f"All {len(result.checks)} source-verification checks passed"
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
            "artifacts": [
                *state.get("artifacts", []),
                ref.model_dump(mode="json"),
                *(r.model_dump(mode="json") for r in faithfulness_refs),
            ],
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


# ===== STAGE-ONLY STUBS =====


def stage_only(name: str):
    """Placeholder node that only records its own name as the stage."""

    def advance(state: GraphState) -> GraphState:
        return {**state, "stage": name}

    return advance
