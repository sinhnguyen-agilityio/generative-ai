"""Build -> Verify -> Fix around the report, plus the reviewer's ``extend`` evidence lookup."""

from __future__ import annotations

import hashlib
import math

from langchain_core.messages import AIMessage

from intelligence_collector.ports import ArtifactStore, VectorIndex
from intelligence_collector.agents.planner import create_todos, plan_record
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
from intelligence_collector.storage.reports import persist_report
from intelligence_collector.agents.worker import _evidence_path
from intelligence_collector.contracts import (
    RepairRoute,
    ResearchBrief,
    ResearchTodo,
    RunStatus,
    Stage,
    TodoStatus,
    VerificationCheck,
    VerificationIssue,
    VerificationResult,
)
from intelligence_collector.graph.nodes.common import (
    EXTENSION_PASSAGES,
    FAITHFULNESS_THRESHOLD,
    MAX_STALLED_ATTEMPTS,
    _activity,
    invalidate_draft_authority,
)
from intelligence_collector.observability.evaluation import (
    ContextPrecisionEvaluator,
    FaithfulnessEvaluator,
    ResponseRelevancyEvaluator,
)
from intelligence_collector.observability.tracing import traced
from intelligence_collector.repair import diagnose, item_key, locate_with_index
from intelligence_collector.storage.evidence_index import ensure_run_indexed
from intelligence_collector.source_verification import (
    _normalize_url,
    citation_errors,
    cited_source_excerpts,
    failure_fingerprint,
    load_catalog,
    save_catalog_state,
)


def _unresolved_lines(verification: dict, limit: int = 3) -> list[str]:
    """Short, user-facing lines for the failed checks of a verification record."""
    lines = []
    for check in (verification or {}).get("check_results", []):
        if check.get("passed"):
            continue
        detail = (check.get("details") or [""])[0]
        label = {
            "coverage": "Coverage",
            "citation": "Citations",
            "faithfulness": "Faithfulness",
        }.get(check.get("gate"), "Check")
        lines.append(f"{label}: {detail[:220]}")
    return lines[:limit]


class BVFNodes:
    """Build -> Verify -> (diagnose and repair) loop around the report.

    Build (``synthesizing``) writes a complete draft. Verify (``verifying``) runs exactly three
    gates on that exact draft: coverage of the required answers and output constraints (LLM
    judge), citation resolution against sources collected in this run, and RAGAS Faithfulness
    strictly greater than ``FAITHFULNESS_THRESHOLD``. PASS leaves the loop at once. FAIL enters ``fixing``, which
    diagnoses the smallest useful correction (see ``repair.py``) and routes to a targeted
    rewrite, Chroma-located evidence, targeted research or a replacement plan. A technical
    evaluator failure is ERROR: it is retried or escalated, never rewritten. At most
    ``recovery_budget`` improvement cycles run; exhaustion or no progress ends in
    ``needs_review``.
    """

    def __init__(
        self,
        vfs: ArtifactStore,
        index: VectorIndex | None,
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
        """The run's catalog: every stored evidence bundle of the active tasks, not narrowed by
        retrieval — used by Verify and quality scoring. Each source holds the passages its
        sub-agent selected from the page (``excerpt``), not the raw page; raw captures stay in
        the VFS and the vector index."""
        return load_catalog(
            self.vfs,
            state["run_id"],
            state.get("evidence", []),
            active_todo_ids=[todo["todo_id"] for todo in state["todos"]]
            if "todos" in state
            else None,
        )

    def _ensure_indexed(self, state) -> None:
        """Before retrieval: re-index the run's stored pages once if the index (e.g. a new
        embedding model) never received them. Retrieval falls back to unfiltered evidence, so
        an index failure here must not stop the run."""
        if self.index is None:
            return
        try:
            ensure_run_indexed(self.vfs, self.index, state["run_id"])
        except Exception:
            pass

    def _writer_catalog(self, state, full, repair):
        """What the writer reads: small corpora directly, a large one narrowed via Chroma; a
        content repair reads the whole un-narrowed catalog (still the selected passages, not raw
        pages), and the Chroma route adds passages located for the affected items."""
        route = (repair or {}).get("route")
        if route in {RepairRoute.VFS, RepairRoute.RESEARCH}:
            return full
        self._ensure_indexed(state)
        narrowed = select_relevant_evidence(
            full,
            self.index,
            state["run_id"],
            top_k=self.evidence_top_k,
            direct_read_chars=self.direct_read_chars,
        )
        if route == RepairRoute.CHROMA and self.index is not None:
            return locate_with_index(
                self.vfs,
                self.index,
                state["run_id"],
                full,
                narrowed,
                repair["issues"],
                state.get("artifacts", []),
            )
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
            {
                "todo_id": bundle.todo_id,
                "question": bundle.question,
                "source_ids": bundle.source_ids,
            }
            for bundle in catalog.bundles
        ]
        ranking_ref = self.vfs.write_json(
            state["run_id"],
            f"synthesis/retrieval.v{version}.json",
            writer_evidence,
            "retrieval",
        )
        return {
            # A changed artifact never inherits a previous verification, approval, publication
            # or quality score.
            **invalidate_draft_authority(),
            "stage": Stage.VERIFYING.value,
            "current_action": (
                f"Draft v{version} repaired ({repair['route']})"
                if repair
                else f"Draft v{version} rewritten per the reviewer's request"
                if reviewing
                else "First draft built from stored evidence"
            ),
            "human_feedback": "",  # consumed
            "activity": _activity(state, "Synthesizing report"),
            "previous_draft": state.get("draft", ""),
            "draft": report,
            "draft_version": version,
            "repair": None,  # consumed
            "writer_evidence": writer_evidence,
            "artifacts": [
                *state.get("artifacts", []),
                ref.model_dump(mode="json"),
                ranking_ref.model_dump(mode="json"),
            ],
        }

    def prepare_revision(self, state):
        """Before a reviewer- or recovery-requested rewrite: fold the reviewer requests into
        the ones still in force and drop the reviewed draft's authority."""
        from intelligence_collector.agents.reviewer_checks import effective_requests

        return {
            **invalidate_draft_authority(),
            "reviewer_requests": effective_requests(list(state.get("reviewer_requests", []))),
        }

    def _error(self, state, message: str, partial: dict | None = None):
        """ERROR: the evaluation could not produce a trustworthy verdict. Nothing is rewritten."""
        return {
            "verification_issue": VerificationIssue.VERIFIER_FAILURE.value,
            "verification": None,
            "verification_outcome": "ERROR",
            "verifier_error": message,
            "last_error": "Report verifier unavailable: " + message,
            "coverage_cache": partial,
            "verifier_attempt": state.get("verifier_attempt", 0) + 1,
        }

    def _faithfulness(self, brief, report, catalog):
        """Third gate. Returns ``(check, score, error)``; ``error`` means ERROR."""
        if self.faithfulness_evaluator is None:
            return (
                None,
                None,
                ("Faithfulness evaluator is not configured" if self.require_faithfulness else None),
            )
        contexts = cited_source_excerpts(report, catalog)
        try:
            with traced(
                "ragas.faithfulness", as_type="evaluator", metadata={"contexts": len(contexts)}
            ) as observation:
                result = self.faithfulness_evaluator(
                    brief.original_question or brief.research_question, report, contexts
                )
                observation.output = (
                    {
                        "score": result.get("score"),
                        "unsupported": len(result.get("unsupported") or []),
                    }
                    if isinstance(result, dict)
                    else {"score": result}
                )
        except Exception as exc:
            return None, None, f"Faithfulness evaluation failed: {type(exc).__name__}"
        score = result.get("score") if isinstance(result, dict) else result
        unsupported = list(result.get("unsupported") or []) if isinstance(result, dict) else []
        if (
            not isinstance(score, (int, float))
            or isinstance(score, bool)
            or math.isnan(score)
            or not 0 <= score <= 1
        ):
            return None, None, "Faithfulness returned no valid score"
        passed = score > FAITHFULNESS_THRESHOLD
        check = VerificationCheck(
            name="faithfulness",
            passed=passed,
            gate="faithfulness",
            details=[]
            if passed
            else [
                f"RAGAS Faithfulness {score:.2f} is not above {FAITHFULNESS_THRESHOLD:.2f}."
                + (" Unsupported statements: " + " | ".join(unsupported[:8]) if unsupported else "")
            ],
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
                    brief,
                    criteria,
                    report,
                    format_evidence(catalog),
                    reviewer_requests=requests,
                    run_id=state["run_id"],
                    thread_id=state.get("thread_id", ""),
                )
            except Exception as exc:
                return self._error(state, str(exc))
            checks = coverage_checks(pairs)
            for item in reviewer_format_checks(report, requests):
                checks.append(
                    VerificationCheck(
                        name="reviewer_word_limit",
                        passed=item.met,
                        gate="coverage",
                        issue_type="presentation",
                        item=item.criterion,
                        details=[] if item.met else [f"{item.criterion} — {item.justification}"],
                    )
                )
            for error in citation_errors(report, catalog):
                checks.append(
                    VerificationCheck(
                        name="citation_resolution",
                        passed=False,
                        gate="citation",
                        issue_type="draft_defect"
                        if error.startswith("Unknown")
                        else "presentation",
                        details=[error],
                        affected_ids=[f"citation:{error}"],
                    )
                )
        score = None
        if all(check.passed for check in checks):
            check, score, error = self._faithfulness(brief, report, catalog)
            if error:
                return self._error(
                    state,
                    error,
                    {
                        "artifact_hash": artifact_hash,
                        "checks": [c.model_dump(mode="json") for c in checks],
                    },
                )
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
            {
                **result.model_dump(mode="json"),
                "outcome": "PASS" if result.passed else "FAIL",
                "draft_version": state.get("draft_version", 0),
                "faithfulness_score": score,
            },
            "verification",
        )
        failed = [check.name for check in result.check_results if not check.passed]
        return {
            "stage": (Stage.AWAITING_APPROVAL if result.passed else Stage.VERIFYING).value,
            "status": (RunStatus.PAUSED if result.passed else RunStatus.ACTIVE).value,
            "current_action": (
                f"All {len(result.checks)} verification checks passed"
                if result.passed
                else f"Verification failed: {', '.join(failed)}"
            ),
            "activity": _activity(state, "Verifying report"),
            "verification": result.model_dump(mode="json"),
            "verification_outcome": "PASS" if result.passed else "FAIL",
            "verification_issue": VerificationIssue.DRAFT_DEFECT.value,
            "faithfulness_score": score,
            "coverage_cache": None,
            "verifier_attempt": 0,
            "verifier_error": "",
            "last_error": "",
            "verification_history": [
                *state.get("verification_history", []),
                result.model_dump(mode="json"),
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
            "context_precision_score": None,
            "context_precision_error": "",
            "context_precision_details": [],
            "response_relevancy_score": None,
            "response_relevancy_error": "",
        }
        # Tell the user a (new) verified report is waiting, so a revision visibly completes.
        update["messages"] = [
            *state.get("messages", []),
            AIMessage(
                content=(
                    f"Report draft v{state.get('draft_version', 1)} passed verification (coverage, "
                    "citations, faithfulness) and is ready for your review below."
                )
            ),
        ]
        if self.context_precision_evaluator is None and self.response_relevancy_evaluator is None:
            return update

        catalog = self._catalog(state)
        question = ResearchBrief.model_validate(state["scoped_request"]).research_question
        draft = state.get("draft", "")
        record: dict = {"question": question, "faithfulness_score": state.get("faithfulness_score")}

        def run(metric: str, evaluate):
            """Run one metric; a failure is recorded under ``<metric>_error``, never raised."""
            try:
                with traced(f"ragas.{metric}", as_type="evaluator") as observation:
                    observation.output = evaluate()
                return observation.output
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
                "context_precision",
                lambda: self.context_precision_evaluator(ranked, draft),
            )
            if result is not None:
                update["context_precision_score"] = result["score"]
                update["context_precision_details"] = result["per_question"]
        if self.response_relevancy_evaluator is not None:
            update["response_relevancy_score"] = run(
                "response_relevancy",
                lambda: self.response_relevancy_evaluator(question, draft),
            )

        verification_id = (state.get("verification") or {}).get("verification_id", "")
        ref = self.vfs.write_json(
            state["run_id"],
            f"verification/quality.{verification_id}.json",
            {**record, **update},
            "quality",
        )
        return {
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
            verification,
            todos=todos,
            catalog=self._catalog(state),
            shown_source_ids=shown,
            researched_items=list(state.get("repair_researched", [])),
            can_replan=self.planner is not None,
        )
        repair["cycle"] = attempt
        history = [
            *state.get("bvf_history", []),
            {
                "attempt": attempt,
                "draft_version": state.get("draft_version", 0),
                "route": repair["route"],
                "failed_checks": [c.name for c in verification.check_results if not c.passed],
                "issues": repair["issues"],
                "fingerprint": state.get("progress_fingerprint", ""),
            },
        ]
        refs = [
            self.vfs.write_json(
                state["run_id"], f"verification/repair.{attempt}.json", repair, "repair_plan"
            ),
            self.vfs.write_json(
                state["run_id"], "verification/bvf_history.json", history, "bvf_history"
            ),
        ]
        return {
            "stage": Stage.SYNTHESIZING.value,
            "recovery_attempt": attempt,
            "repair": repair,
            "current_action": f"Improvement cycle {attempt}/{budget}: {repair['route']} repair",
            "activity": _activity(state, f"Cycle {attempt}: {repair['route']} repair"),
            "bvf_history": history,
            "artifacts": [
                *state.get("artifacts", []),
                *(ref.model_dump(mode="json") for ref in refs),
            ],
        }

    def prepare_repair_research(self, state):
        """Reopen only the affected approved tasks for targeted Tavily research on the missing
        points; completed evidence stays and other tasks are not researched again."""
        repair = state.get("repair") or {}
        targets: dict[str, list[str]] = {}
        for issue in repair.get("issues", []):
            if issue["route"] == RepairRoute.RESEARCH and issue["todo_id"]:
                targets.setdefault(issue["todo_id"], []).extend(
                    issue["missing_points"] or [issue["item"]]
                )
        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        if not targets or not set(targets) <= {todo.todo_id for todo in todos}:
            raise ValueError("Targeted research requires known affected task ids")
        for todo in todos:
            if todo.todo_id in targets:
                todo.status = TodoStatus.PLANNED
                todo.research_attempt += 1
                todo.research_instruction = "Find evidence for: " + "; ".join(targets[todo.todo_id])
        researched = [
            *state.get("repair_researched", []),
            *(
                item_key(issue["item"])
                for issue in repair.get("issues", [])
                if issue["route"] == RepairRoute.RESEARCH
            ),
        ]
        return {
            "todos": [todo.model_dump(mode="json") for todo in todos],
            "repair_researched": list(dict.fromkeys(researched)),
            # Targeted research replaces, not adds to, the pre-draft gap allowance.
            "research_gap_attempt": max(state.get("research_gap_attempt", 0), 1),
            "stage": Stage.RESEARCHING.value,
            "current_action": f"Targeted research for {len(targets)} task(s)",
        }

    def replan(self, state):
        """Propose replacement tasks and wait for plan approval, keeping the original checklist
        and completed evidence. Reached when targeted research did not help (repair route
        "replan") or when the research supervisor proposed a replan."""
        if self.planner is None:
            raise ValueError("Replacement planning requires a Planner")
        brief = ResearchBrief.model_validate(state["scoped_request"])
        repair = state.get("repair") or {}
        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        proposal = state.get("research_replan") or {}
        questions = {todo.todo_id: todo.research_question for todo in todos}
        unanswered = list(
            dict.fromkeys(
                [
                    issue["item"]
                    for issue in repair.get("issues", [])
                    if issue["route"] == RepairRoute.REPLAN
                ]
                + [
                    f"{questions[todo_id]} ({proposal.get('reason', '')})"
                    for todo_id in proposal.get("todo_ids", [])
                    if todo_id in questions
                ]
            )
        )
        plan = self.planner.define_plan(
            brief,
            unanswered=unanswered,
            previous_questions=[todo.research_question for todo in todos],
            run_id=state["run_id"],
            thread_id=state.get("thread_id", ""),
        )
        new = create_todos(self.vfs, state["run_id"], plan.questions, plan.evidence_needed)
        record = plan_record(brief, [*todos, *new], plan.report_shape, kind="replacement")
        version = state.get("plan_version", 0) + 1
        ref = self.vfs.write_json(
            state["run_id"], f"planning/plan.v{version}.json", record, "research_plan"
        )
        return {
            "todos": [todo.model_dump(mode="json") for todo in [*todos, *new]],
            "plan": record,
            "plan_version": version,
            "plan_approval": None,
            "repair": {**repair, "route": RepairRoute.RESEARCH.value},
            "research_replan": None,
            "stage": Stage.PLANNING.value,
            "current_action": "Replacement plan awaiting approval",
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }

    def _extension_passages(self, state, catalog, topic: str) -> list:
        """Sources of this run's active evidence that the vector index ranks closest to `topic`,
        across all research tasks, at most ``EXTENSION_PASSAGES``. Hits that do not map to a
        source of the active catalog (an old scope, another run) are ignored."""
        if self.index is None:
            return []
        self._ensure_indexed(state)
        try:
            chunks = self.index.retrieve(topic, limit=EXTENSION_PASSAGES, run_id=state["run_id"])
        except Exception:
            return []
        by_url = {_normalize_url(source.url): sid for sid, source in catalog.sources.items()}
        found = dict.fromkeys(
            sid
            for chunk in chunks
            if (sid := by_url.get(_normalize_url(chunk.metadata.get("locator", ""))))
        )
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
                    brief.original_question or brief.research_question,
                    request,
                    topic,
                    passages,
                    run_id=run_id,
                    thread_id=state.get("thread_id", ""),
                )
            except Exception as exc:
                judge_error = f"{type(exc).__name__}: extension judge failed"
        offered = {source.source_id for source in passages}
        relevant = [
            sid
            for sid in dict.fromkeys(verdict.relevant_source_ids if verdict else [])
            if sid in offered
        ]
        # The judge's "sufficient" counts only with at least one real passage behind it.
        sufficient = bool(verdict and verdict.sufficient and relevant)
        missing = [
            point.strip() for point in (verdict.missing_points if verdict else []) if point.strip()
        ]

        todos = [ResearchTodo.model_validate(todo) for todo in state.get("todos", [])]
        evidence = list(state.get("evidence", []))
        artifacts = list(state.get("artifacts", []))
        if sufficient:
            todo = ResearchTodo(
                research_question=topic, evidence_needed=request, status=TodoStatus.COMPLETED
            )
            bundle_ref = self.vfs.write_json(
                run_id,
                _evidence_path(todo),
                {
                    "task_id": todo.todo_id,
                    "research_attempt": todo.research_attempt,
                    "question": topic,
                    "summary": verdict.summary.strip() or f"Stored evidence on: {topic}",
                    "sources": [
                        {
                            "locator": catalog.sources[sid].url,
                            "provenance": {"title": catalog.sources[sid].title},
                            "excerpt": catalog.sources[sid].excerpt,
                        }
                        for sid in relevant
                    ],
                    "sufficient": True,
                    "unresolved_gaps": [],
                    "indexed_chunks": [],
                    "located_from_index": True,
                },
                "evidence_bundle",
                task_id=todo.todo_id,
            )
            evidence.append(bundle_ref.model_dump(mode="json"))
            artifacts.append(bundle_ref.model_dump(mode="json"))
        else:
            todo = ResearchTodo(
                research_question=topic,
                evidence_needed=request,
                research_instruction="; ".join(missing),
            )
        todos.append(todo)

        update: dict = {}
        if state.get("plan") is not None:
            # The reviewer's request is the approval of this one added task.
            plan = plan_record(
                brief, todos, state["plan"].get("report_shape", ""), kind="extension"
            )
            version = state.get("plan_version", 0) + 1
            plan_ref = self.vfs.write_json(
                run_id, f"planning/plan.v{version}.json", plan, "research_plan"
            )
            artifacts.append(plan_ref.model_dump(mode="json"))
            update = {
                "plan": plan,
                "plan_version": version,
                "plan_approval": {
                    "plan_hash": plan["plan_hash"],
                    "approved": True,
                    "outcome": "approve",
                    "actor": state.get("reviewer") or "local-user",
                    "kind": "extension",
                },
            }

        record = {
            "topic": topic,
            "request": request,
            "todo_id": todo.todo_id,
            "sufficient": sufficient,
            "passages": sorted(offered, key=lambda sid: int(sid[1:])),
            "relevant_source_ids": relevant,
            "missing_points": missing,
            "judge_error": judge_error,
        }
        count = sum(1 for item in state.get("bvf_history", []) if item.get("kind") == "EXTENSION")
        artifacts.append(
            self.vfs.write_json(
                run_id, f"extension/{count + 1}.json", record, "extension"
            ).model_dump(mode="json")
        )
        action = (
            f"Stored evidence covers the new section ({len(relevant)} source(s))"
            if sufficient
            else f"Researching the new section: {topic}"
        )
        return {
            **update,
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
        reason = (
            state.get("last_error")
            if state.get("verification_issue")
            in {
                VerificationIssue.VERIFIER_FAILURE,
                VerificationIssue.RESEARCH_FAILURE,
                VerificationIssue.PUBLICATION_FAILURE,
            }
            else (
                "replacement plan declined"
                if (state.get("plan_approval") or {}).get("outcome") == "decline"
                else "research gap budget exhausted"
                if state.get("research_gap_attempt", 0) >= state.get("research_gap_budget", 2)
                and not state.get("draft")
                else "no progress after repair"
                if state.get("stall_count", 0) >= MAX_STALLED_ATTEMPTS
                else f"improvement cycle budget {state.get('recovery_budget', self.recovery_budget)} exhausted"
            )
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
        notice = (
            f"Report draft v{version} could not be verified ({reason}). It is shown below "
            "as an unverified draft; choose a recovery option to continue."
            if state.get("draft")
            else f"No report could be verified ({reason}). Choose a recovery option to continue."
        )
        notice += "".join(f"\n- {line}" for line in _unresolved_lines(verification))
        return {
            "stage": Stage.NEEDS_REVIEW.value,
            "status": RunStatus.NEEDS_REVIEW.value,
            "messages": [*state.get("messages", []), AIMessage(content=notice)],
            "last_error": verification.get("repair_details") or state.get("last_error", ""),
            "current_action": f"Needs review: {reason} after {attempts} improvement cycle(s)",
            "activity": _activity(state, "Verification needs review", "failure"),
            "artifacts": [*state.get("artifacts", []), ref.model_dump(mode="json")],
        }
