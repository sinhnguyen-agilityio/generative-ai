"""Durable policy and focused verification/diagnosis components.

VFS records bridge operation/checkpoint crash windows. A single process owns a run.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from intelligence_collector.contracts import (
    ResearchBrief, ReportJudgeVerdict, VerificationResult, VerificationCheck,
)
from intelligence_collector.models.router import _is_transient
from intelligence_collector.models.redaction import redact_text, redact_value
from intelligence_collector.agents.report_writer import format_evidence
from intelligence_collector.agents.reviewer_checks import reviewer_format_checks
from intelligence_collector.source_verification import citation_errors


def digest(value):
    data = value if isinstance(value, str) else json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(data.encode()).hexdigest()


def operation_progress(operation, key, status, inputs):
    """Publish worker progress through LangGraph; workers never call Streamlit."""
    from langgraph.config import get_stream_writer
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer({"kind": "operation", "operation": operation, "id": key,
            "status": status, "details": redact_value({k: inputs[k] for k in
                ("url", "query", "task_id") if k in inputs})})


def versions(state, vfs=None):
    evidence = []
    for ref in state.get("evidence", []):
        content_hash = ref.get("content_hash")
        if vfs is not None:
            relative = str(Path(ref["path"]).resolve().relative_to((vfs.root / state["run_id"]).resolve()))
            content_hash = hashlib.sha256(vfs.read(state["run_id"], relative)).hexdigest()
        evidence.append((ref.get("path"), content_hash))
    return {
        "artifact_hash": digest(state.get("draft", "")),
        "requirements_hash": digest({
            "brief": state.get("scoped_request"), "checklist": [
                {"todo_id": t["todo_id"], "question": t["research_question"]}
                for t in state.get("required_checklist", state.get("todos", []))
            ], "requests": state.get("reviewer_requests", []),
            "guidelines": state.get("report_guidelines", []),
        }),
        "evidence_hash": digest(evidence),
    }


class VerificationOutputError(ValueError):
    """Incomplete evaluation, optionally retaining trustworthy report assessments."""

    def __init__(self, message, *, partial_results=()):
        super().__init__(message)
        self.partial_results = list(partial_results)


def verification_scope(state):
    """A deterministic scope per checkpointed verifier attempt and explicit resume."""
    attempt = state.get("verifier_attempt", 0)
    epoch = state.get("verifier_retry_epoch", 0)
    if not attempt and not epoch:
        return None
    return {"epoch": epoch, "attempt": attempt + 1,
            "previous_error": state.get("verifier_error", "")}


class OperationFailure(RuntimeError):
    def __init__(self, record):
        super().__init__(record.get("error", "Operation failed"))
        self.record = record


class DurableOperations:
    """Exactly one retry owner for non-model operations, with durable attempt records."""

    def __init__(self, vfs):
        self.vfs = vfs

    def call(self, run_id, operation, inputs, invoke, *, attempts=2, retry_scope=None, validate_result=None):
        key = digest({"operation": operation, "inputs": inputs})
        relative = f"operations/policy/{key}.json"
        try:
            record = json.loads(self.vfs.read(run_id, relative))
        except FileNotFoundError:
            record = {"operation": operation, "attempts": 0, "status": "pending"}
        def cached_result():
            try:
                if validate_result is not None:
                    validate_result(record["result"])
            except Exception as exc:
                # A legacy/invalid cached verdict must never bypass gate validation.
                record.update(status="error", error=redact_text(exc), retriable=False,
                              partial_results=getattr(exc, "partial_results", []))
                self.vfs.write_json(run_id, relative, record, "operation_invalid_result")
                raise OperationFailure(record) from exc
            return record["result"]
        if record["status"] == "complete":
            return cached_result()
        if retry_scope is not None:
            # Reuse completed gates above; retain failed records and allocate a scoped attempt.
            key = digest({"operation": operation, "inputs": inputs, "retry_scope": retry_scope})
            relative = f"operations/policy/{key}.json"
            try:
                record = json.loads(self.vfs.read(run_id, relative))
            except FileNotFoundError:
                record = {"operation": operation, "attempts": 0, "status": "pending"}
            if record["status"] == "complete":
                return cached_result()
        if record["attempts"] >= attempts or record.get("retriable") is False:
            raise OperationFailure(record)
        while record["attempts"] < attempts:
            # Save intent before the call. An abandoned attempt still consumes allowance.
            record.update(attempts=record["attempts"] + 1, status="running")
            self.vfs.write_json(run_id, relative, record, "operation_attempt")
            operation_progress(operation, key, "running", inputs)
            try:
                from intelligence_collector.observability.tracing import traced
                with traced(f"operation.{operation}",
                            input={"arguments": redact_value(inputs)},
                            metadata={"operation_id": key, "attempt": record["attempts"]}):
                    result = invoke()
                if validate_result is not None:
                    validate_result(result)
                if hasattr(result, "model_dump"):
                    result = result.model_dump(mode="json")
            except Exception as exc:
                operation_progress(operation, key, "failure", inputs)
                record.update(status="error", error=redact_text(exc), retriable=_is_transient(exc),
                              partial_results=getattr(exc, "partial_results", []))
                self.vfs.write_json(run_id, relative, record, "operation_attempt")
                if not record["retriable"] or record["attempts"] >= attempts:
                    raise OperationFailure(record) from exc
            else:
                operation_progress(operation, key, "complete", inputs)
                record.update(status="complete", result=result, error="")
                self.vfs.write_json(run_id, relative, record, "operation_result")
                return result
        raise OperationFailure(record)


class ImprovementLedger:
    """Stable batch/cycle IDs. Only explicit feedback opens another batch."""

    def __init__(self, vfs):
        self.vfs = vfs

    def load(self, state):
        try:
            return json.loads(self.vfs.read(state["run_id"], "workflow/improvements.json"))
        except FileNotFoundError:
            return {
                "batch_id": state.get("batch_id", "initial"),
                "batch_cycles": state.get("batch_cycles", state.get("recovery_attempt", 0)),
                "cumulative_cycles": state.get("cumulative_cycles", state.get("recovery_attempt", 0)),
                "active_cycle_id": state.get("active_cycle_id", ""),
                "completed_cycle_ids": state.get("completed_cycle_ids", []),
                "feedback_submission_ids": state.get("feedback_submission_ids", []),
            }

    def save(self, state, value):
        self.vfs.write_json(state["run_id"], "workflow/improvements.json", value, "improvement_ledger")
        return value

    def open_feedback(self, state, submission_id):
        if not submission_id:
            raise ValueError("A new explicit feedback submission needs a stable ID")
        value = self.load(state)
        if submission_id in value["feedback_submission_ids"]:
            return value
        value.update(batch_id=f"feedback:{submission_id}", batch_cycles=0, active_cycle_id="")
        value["feedback_submission_ids"].append(submission_id)
        return self.save(state, value)

    def begin(self, state, budget):
        value = self.load(state)
        if value["active_cycle_id"]:
            return value
        if value["batch_cycles"] >= budget:
            raise ValueError("Improvement batch exhausted")
        value["active_cycle_id"] = f'{value["batch_id"]}:{value["batch_cycles"] + 1}'
        return self.save(state, value)

    def finish(self, state):
        value = self.load(state)
        cycle = value["active_cycle_id"]
        if cycle and cycle not in value["completed_cycle_ids"]:
            value["completed_cycle_ids"].append(cycle)
            value["batch_cycles"] += 1
            value["cumulative_cycles"] += 1
        value["active_cycle_id"] = ""
        return self.save(state, value)


class ReportVerification:
    """Exactly coverage, citation resolution, and strictly > .90 Faithfulness."""

    def __init__(self, vfs, judge, faithfulness, guidelines):
        self.vfs, self.judge, self.faithfulness = vfs, judge, faithfulness
        self.guidelines = list(guidelines)
        self.operations = DurableOperations(vfs)

    def verify(self, state, catalog):
        state = {**state, "report_guidelines": self.guidelines}
        binding = versions(state, self.vfs)
        brief = ResearchBrief.model_validate(state["scoped_request"])
        criteria = [
            f"Answer required original question: {q}" for q in
            (brief.original_questions or [brief.research_question])
        ]
        criteria += [f"Answer required checklist item {t['todo_id']}: {t['research_question']}"
                     for t in state.get("required_checklist", state.get("todos", []))]
        criteria += [*brief.constraints, *brief.output_requirements, *brief.success_criteria,
                     *self.guidelines, *state.get("reviewer_requests", [])]
        criteria = list(dict.fromkeys(criteria))
        criteria.append("Evidence gaps may satisfy an item only under gap_policy=disclosed; "
                        f"the applicable gap_policy is {brief.gap_policy}.")
        checks, issues = [], []
        scope = verification_scope(state)
        def validate_coverage(raw):
            try:
                verdict = ReportJudgeVerdict.model_validate(raw)
            except ValueError as exc:
                raise VerificationOutputError("Invalid coverage evaluation schema") from exc
            from collections import Counter
            if Counter(r.criterion for r in verdict.results) != Counter(criteria):
                raise VerificationOutputError("Partial or invalid coverage evaluation: every criterion is required exactly once")

        def coverage_call():
            return self.judge.verify(
                brief, criteria, state["draft"], format_evidence(catalog),
                reviewer_requests=state.get("reviewer_requests", []),
                run_id=state["run_id"], thread_id=state.get("thread_id", ""))
        try:
            verdict = ReportJudgeVerdict.model_validate(self.operations.call(
                state["run_id"], "coverage", binding, coverage_call,
                attempts=1, retry_scope=scope, validate_result=validate_coverage,
                # ModelRouter owns transport retry/fallback.
            ))
            failures = [r for r in verdict.results if not r.met]
            formatting = reviewer_format_checks(state["draft"], state.get("reviewer_requests", []))
            details = [r.justification for r in failures] + [
                r.justification for r in formatting if not r.met]
            checks.append(VerificationCheck(name="coverage", passed=not details, details=details))
            for r in failures:
                # Diagnosis resolves these candidates using independent stored evidence.
                issues.append({"question": r.criterion, "location": r.criterion,
                               "reason": r.justification, "issue_type": r.issue_type,
                               "evidence_refs": [], "route": ""})
            for r in formatting:
                if not r.met:
                    issues.append({"question": "", "location": r.criterion,
                                   "reason": r.justification, "issue_type": "presentation",
                                   "evidence_refs": [], "route": "presentation"})
            citation = citation_errors(state["draft"], catalog)
            checks.append(VerificationCheck(name="citation_resolution", passed=not citation, details=citation))
            issues += [{"question": "", "location": "Sources", "reason": e,
                        "issue_type": "presentation", "evidence_refs": [],
                        "route": "presentation"} for e in citation]
            if self.faithfulness is None:
                raise ValueError("RAGAS Faithfulness evaluator is required")
            def score_call():
                score = self.faithfulness(brief.research_question, state["draft"],
                                         [s.excerpt for s in catalog.sources.values() if s.excerpt])
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise VerificationOutputError("Missing or invalid RAGAS Faithfulness score")
                return score
            score = self.operations.call(state["run_id"], "faithfulness", binding, score_call,
                attempts=1 if getattr(self.faithfulness, "retry_owner", "") == "ragas" else 2,
                retry_scope=scope)
            checks.append(VerificationCheck(name="ragas_faithfulness", passed=score > .90,
                                             details=[] if score > .90 else [f"Faithfulness {score} must exceed 0.90"]))
            if score <= .90:
                # Location-specific diagnosis must happen before rewriting. Never just send a score.
                for section in state["draft"].split("\n## "):
                    if section.strip() and not section.lower().startswith(("sources", "scope", "limitations")):
                        issues.append({"question": "", "location": section.splitlines()[0],
                                       "reason": "Inspect this section against original evidence; "
                                                 f"report Faithfulness={score} is below threshold",
                                       "issue_type": "faithfulness", "evidence_refs": list(catalog.sources),
                                       "route": "vfs"})
            outcome = "PASS" if all(c.passed for c in checks) else "FAIL"
            result = VerificationResult(**binding, outcome=outcome, passed=outcome == "PASS",
                                        checks=[c.name for c in checks], check_results=checks,
                                        issues=issues, repair_details="\n".join(
                                            f"{i['location']}: {i['reason']}" for i in issues))
            return result, score, None
        except Exception as exc:
            record = getattr(exc, "record", {"operation": "verification", "attempts": 1,
                                            "error": redact_text(exc), "retriable": False})
            # Known failures remain actionable even when another gate cannot finish.
            from intelligence_collector.contracts import ReportJudgeCriterion
            from collections import Counter
            partial = record.get("partial_results", [])
            partial_counts = Counter(item.get("criterion") for item in partial if isinstance(item, dict))
            for raw in partial:
                try:
                    item = ReportJudgeCriterion.model_validate(raw)
                except ValueError:
                    continue
                if item.criterion not in criteria or partial_counts[item.criterion] != 1 or item.met:
                    continue
                issues.append({"question": item.criterion, "location": item.criterion,
                               "reason": item.justification, "issue_type": item.issue_type,
                               "evidence_refs": [], "route": ""})
            record = {**record, "repairable": bool(issues)}
            # Partial gate results are diagnostic only; ERROR cannot pass.
            result = VerificationResult(**binding, outcome="ERROR", passed=False,
                                        checks=["coverage", "citation_resolution", "ragas_faithfulness"],
                                        check_results=checks, issues=issues, repair_details="\n".join(
                                            [*(f"{i['location']}: {i['reason']}" for i in issues), redact_text(exc)]))
            return result, None, record


class RepairDiagnosis:
    """Select the narrowest corrective route, checking original VFS after index retrieval."""

    def __init__(self, vfs, index, router=None):
        self.vfs, self.index, self.router = vfs, index, router

    def diagnose(self, state, catalog):
        candidates = (state.get("verification") or {}).get("issues", [])
        if state.get("human_feedback"):
            candidates = [{"question": "", "location": "Human feedback",
                           "reason": state["human_feedback"], "issue_type": "feedback",
                           "evidence_refs": [], "route": ""}]
        if not candidates:
            candidates = [{"location": "Report", "question": "",
                           "reason": (state.get("verification") or {}).get("repair_details", ""),
                           "issue_type": "draft_defect", "evidence_refs": [], "route": ""}]
        if self.router is not None:
            from langchain_core.messages import SystemMessage, HumanMessage
            from intelligence_collector.contracts import CorrectiveDiagnosis
            verdict = self.router.invoke_messages("fixer", [
                SystemMessage(content="Diagnose before repair. Return actionable issues with the affected "
                    "existing question/todo ID, exact report location, reason, evidence refs, and route. "
                    "Use vfs when identifiable collected originals suffice; chroma when potentially indexed; "
                    "research only for missing evidence within the approved strategy; presentation for output "
                    "constraints; replan for changed direction, source categories or decomposition. A replacement "
                    "strategy must specify the proposed changes in reason and requires human approval. Never drop "
                    "original questions. Diagnose low Faithfulness against original evidence, not just the score."),
                HumanMessage(content=json.dumps({"issues": candidates, "report": state.get("draft"),
                    "brief": state.get("scoped_request"), "plan": state.get("plan"),
                    "todos": state.get("todos"), "original_evidence": format_evidence(catalog)})),
            ], CorrectiveDiagnosis, run_id=state["run_id"], thread_id=state.get("thread_id", ""))
            candidates = [i.model_dump(mode="json") for i in verdict.issues]
        issues = []
        for candidate in candidates:
            issue = {**candidate, "report_hash": digest(state.get("draft", ""))}
            text = (issue.get("question", "") + " " + issue["reason"]).lower()
            if issue.get("route") == "replan" or any(w in text for w in ("replacement strategy", "change source categories",
                                      "change research direction", "task decomposition")):
                issue["route"] = "replan"
            elif issue.get("route") == "presentation" or any(
                w in text for w in ("format", "language", "word limit", "shorter", "markdown",
                                   "structure", "tiếng", "length")):
                issue["route"] = "presentation"
            else:
                relevant = [b for b in catalog.bundles if b.source_ids and (
                    b.todo_id in text or any(word in text for word in b.question.lower().split() if len(word) > 4))]
                if issue.get("route") == "research":
                    issue["route"] = "research"
                elif relevant or issue.get("route") == "vfs":
                    issue["route"] = "vfs"
                    issue["evidence_refs"] = list(dict.fromkeys(
                        sid for b in relevant for sid in b.source_ids)) or list(catalog.sources)
                else:
                    chunks = self.index.retrieve(text, run_id=state["run_id"], limit=5)
                    refs = []
                    for chunk in chunks:
                        relative = chunk.metadata.get("vfs_path")
                        if not relative:
                            # Older index records: resolve source provenance via authoritative bundles.
                            relative = next((str(Path(r["path"]).resolve().relative_to(
                                (self.vfs.root / state["run_id"]).resolve()))
                                for r in state.get("artifacts", [])
                                if r.get("artifact_id") == chunk.artifact_id), None)
                        if relative:
                            self.vfs.read(state["run_id"], relative)
                            refs.append(relative)
                    issue["evidence_refs"] = refs
                    issue["route"] = "chroma" if refs else "research"
            issues.append(issue)
        priority = ("replan", "research", "chroma", "vfs", "presentation")
        route = next(r for r in priority if any(i["route"] == r for i in issues))
        return route, issues
