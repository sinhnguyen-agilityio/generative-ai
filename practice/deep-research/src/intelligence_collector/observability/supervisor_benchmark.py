"""Metrics of the supervisor–searcher benchmark (spec 002, SC-002, SC-004, SC-005, SC-007, SC-008).

Each ``score_*`` function runs one component on its labeled fixture in
``evals/supervisor_searcher/`` with the router it is given (scripted in tests, live in
``scripts/benchmark_supervisor_searcher.py``) and returns plain counts and rates. ``compare``
sets the end-to-end figures of a run against ``baseline.json``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from intelligence_collector.agents.claim_judge import ClaimJudge
from intelligence_collector.agents.gap_analysis import GapAnalyzer
from intelligence_collector.agents.result_evaluation import evaluate_results
from intelligence_collector.contracts import (
    EvidencePoint,
    IndexedChunk,
    ResearchTodo,
    SearcherReport,
)
from intelligence_collector.ports import ModelRouterPort
from intelligence_collector.storage.passage_lookup import PassageLookup
from intelligence_collector.storage.vector_index import LocalVectorIndex

FIXTURES = Path(__file__).resolve().parents[3] / "evals" / "supervisor_searcher"
RUN_ID = "bench"


def load_fixture(name: str, root: Path = FIXTURES) -> dict:
    return json.loads((root / name).read_text(encoding="utf-8"))


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score_result_evaluation(router: ModelRouterPort, fixture: dict) -> dict:
    """SC-002: precision and recall of "accepted" against the rows labeled ``relevant``."""
    accepted_relevant = accepted = relevant = 0
    for case in fixture["cases"]:
        todo = ResearchTodo(
            todo_id=case["id"],
            research_question=case["task_question"],
            evidence_points=[
                EvidencePoint(point_id=p["point_id"], text=p["text"])
                for p in case["evidence_points"]
            ],
        )
        rows = [{k: v for k, v in row.items() if k not in {"label", "covers"}} for row in case["rows"]]
        outcome = evaluate_results(
            rows,
            todo=todo,
            question=case["task_question"],
            query=case["task_question"],
            search_index=1,
            seen_urls=set(),
            router=router,
            run_id=RUN_ID,
        )
        for row, decision in zip(case["rows"], outcome.evaluations):
            is_relevant = row["label"] == "relevant"
            is_accepted = decision.decision == "accepted"
            relevant += is_relevant
            accepted += is_accepted
            accepted_relevant += is_relevant and is_accepted
    return {
        "rows": sum(len(case["rows"]) for case in fixture["cases"]),
        "precision": _rate(accepted_relevant, accepted),
        "recall": _rate(accepted_relevant, relevant),
    }


def _case_todos(case: dict) -> list[ResearchTodo]:
    checklist = case["required_answers"]
    return [
        ResearchTodo(
            todo_id=task["todo_id"],
            research_question=task["question"],
            evidence_points=[
                EvidencePoint(
                    point_id=p["point_id"],
                    text=p["text"],
                    required_answers=[checklist[i - 1] for i in p["required_answers"]],
                )
                for p in task["evidence_points"]
            ],
        )
        for task in case["tasks"]
    ]


def score_gap_analysis(router: ModelRouterPort, fixture: dict) -> dict:
    """SC-004: share of planted essential gaps closed in the round (covered by a stored page or
    given a follow-up task), and follow-ups created where no essential gap was planted."""
    planted = closed = follow_ups_without_gap = 0
    for case in fixture["cases"]:
        expected = set(case["expected"]["essential_gaps"])
        pages = {p["source_id"]: p["text"] for p in case["pages"]}
        with tempfile.TemporaryDirectory() as scratch:
            index = LocalVectorIndex(Path(scratch) / "index.json")
            if case["pages"]:
                index.index([
                    IndexedChunk(
                        chunk_id=f"{RUN_ID}:{p['source_id']}:0",
                        source_id=p["source_id"],
                        artifact_id=p["source_id"],
                        text=p["text"],
                        metadata={"run_id": RUN_ID, "todo_id": p["todo_id"], "locator": p["url"]},
                    )
                    for p in case["pages"]
                ])
            result = GapAnalyzer(router, PassageLookup(index)).analyze(
                run_id=RUN_ID,
                todos=_case_todos(case),
                reports={
                    r["todo_id"]: SearcherReport.model_validate({**r, "stop_reason": "no_progress"})
                    for r in case["reports"]
                },
                checklist=case["required_answers"],
                round_number=1,
                previous=None,
                stall_count=0,
                allowed_source_ids=set(pages),
                stored_text=lambda sid, pages=pages: pages.get(sid, ""),
            )
        followed = {key for todo in result.follow_ups for key in todo.gap_point_ids}
        covered = {
            g.point_key
            for g in result.analysis.gaps
            if g.resolution in {"covered_by_report", "covered_by_lookup"}
        }
        planted += len(expected)
        closed += len(expected & (covered | followed))
        follow_ups_without_gap += sum(
            not set(todo.gap_point_ids) & expected for todo in result.follow_ups
        )
    return {
        "planted_essential_gaps": planted,
        "closed": closed,
        "closure_rate": _rate(closed, planted),
        "follow_ups_without_gap": follow_ups_without_gap,
    }


def score_claims(router: ModelRouterPort, fixture: dict) -> dict:
    """SC-005: planted claim errors the gate catches with their own failure code, and correct
    sentences it flags anyway."""
    sentences = [s for s in fixture["sentences"] if s["label"] != "analysis"]
    body = "\n".join(s["sentence"] for s in sentences)
    report = f"# Benchmark\n\n## Findings\n{body}\n"
    evaluation = ClaimJudge(router).evaluate(
        report, fixture["snapshot"], draft_version=1, artifact_hash="bench", run_id=RUN_ID
    )
    by_sentence = {claim.sentence: claim for claim in evaluation.claims}
    planted = detected = correct = false_flags = 0
    for item in sentences:
        claim = by_sentence.get(item["sentence"])
        failures = claim.failures if claim is not None else []
        if item["failure"] is None:
            correct += 1
            false_flags += bool(failures)
        else:
            planted += 1
            detected += any(f.startswith(item["failure"]) for f in failures)
    return {
        "planted_errors": planted,
        "detected": detected,
        "detection_rate": _rate(detected, planted),
        "correct_sentences": correct,
        "false_flags": false_flags,
    }


def compare(summary: dict, baseline: dict | None) -> dict:
    """SC-007 and SC-008 against ``baseline.json``: repair-cycle share and mean searches per
    task, each with the baseline value and the change (None when either side is missing)."""
    before = (baseline or {}).get("summary") or {}
    figures = {}
    for key in ("repair_cycle_share", "mean_searches_per_task"):
        now, then = summary.get(key), before.get(key)
        figures[key] = {
            "baseline": then,
            "now": now,
            "change": now - then if now is not None and then is not None else None,
        }
    return figures
