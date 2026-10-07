"""Opt-in live benchmark for the supervisor–searcher workflow (spec 002, SC-002…SC-008).

Runs every question of ``evals/supervisor_searcher/questions.json`` end to end through the app's
workflow: a clarification pause is recorded as ``skipped``, plan approval is resumed with
``{"approved": true}``, and the run stops at the publication-approval (or recovery) pause after
verification. Nothing is ever published.

``--baseline`` writes ``evals/supervisor_searcher/baseline.json`` (run it on the unchanged code);
otherwise the results go to ``evals/supervisor_searcher/benchmark_results.json``. Each file holds
per-question records and a summary: the share of runs whose draft needed at least one repair cycle
(SC-007) and the mean web searches per task (SC-008). ``benchmark_results.json`` also scores the
live components on the labeled fixtures (result evaluation SC-002, gap analysis SC-004, claim
judge SC-005; see ``intelligence_collector.observability.supervisor_benchmark``) and compares
SC-007 and SC-008 with ``baseline.json``. Errors are stored redacted, and the whole file passes
through ``redact_value`` before it is written.

Requires ``--run-live``: model, embedding and Tavily calls are paid.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BENCH_DIR = ROOT / "evals" / "supervisor_searcher"
QUESTIONS = BENCH_DIR / "questions.json"
# Waiting kinds that end a benchmark run: verification has happened (or cannot happen).
TERMINAL_INTERRUPTS = {"publication_approval", "recovery_required"}
MAX_RESUMES = 4


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _interrupt_kind(snapshot) -> str:
    for task in getattr(snapshot, "tasks", ()) or ():
        for interrupt in getattr(task, "interrupts", ()) or ():
            value = getattr(interrupt, "value", None)
            if isinstance(value, dict):
                return str(value.get("kind") or "")
    return ""


def searches_per_task(vfs_root: Path, run_id: str) -> list[int]:
    """Web searches each research task spent, read from its stored evidence bundles."""
    counts = []
    for path in sorted((vfs_root / run_id).glob("research/**/evidence/result.json")):
        try:
            bundle = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(bundle.get("search_calls"), int):
            counts.append(bundle["search_calls"])
    return counts


def write_result(target: Path, result: dict, redact_value) -> None:
    """Write `result` as JSON with every credential-shaped value masked."""
    target.write_text(
        json.dumps(redact_value(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def summarize(records: list[dict]) -> dict:
    """SC-007 and SC-008 figures over the runs that reached verification."""
    verified = [r for r in records if r.get("reached_verification")]
    searches = [n for r in records for n in r.get("searches_per_task", [])]
    return {
        "questions": len(records),
        "reached_verification": len(verified),
        "repair_cycle_share": (
            sum(1 for r in verified if r.get("repair_cycles", 0) > 0) / len(verified)
            if verified
            else None
        ),
        "mean_searches_per_task": sum(searches) / len(searches) if searches else None,
    }


def run_question(graph, vfs_root: Path, question: dict, redact) -> dict:
    """Drive one question to its verification pause and return its record."""
    from langgraph.types import Command

    run_id = str(uuid4())
    config = {"configurable": {"thread_id": run_id}, "recursion_limit": 150}
    record: dict = {"id": question["id"], "run_id": run_id, "outcome": "", "error": ""}
    started = time.monotonic()
    payload = {"run_id": run_id, "thread_id": run_id, "request": question["question"]}
    try:
        for _ in range(MAX_RESUMES + 1):
            graph.invoke(payload, config)
            snapshot = graph.get_state(config)
            kind = _interrupt_kind(snapshot)
            if kind == "plan_approval":
                payload = Command(resume={"approved": True})
                continue
            if kind == "clarification_required":
                record["outcome"] = "skipped"
            else:
                record["outcome"] = kind or "ended"
            break
        else:
            record["outcome"] = "resume_limit"
    except Exception as exc:  # a failed question must not stop the benchmark
        record["outcome"] = "error"
        record["error"] = f"{type(exc).__name__}: {redact(exc)}"
    state = graph.get_state(config).values
    verification = state.get("verification") or {}
    record.update(
        stage=state.get("stage", ""),
        reached_verification=record["outcome"] in TERMINAL_INTERRUPTS,
        verified=bool(verification.get("passed")),
        repair_cycles=int(state.get("recovery_attempt", 0) or 0),
        draft_version=int(state.get("draft_version", 0) or 0),
        tasks=len(state.get("todos", []) or []),
        searches_per_task=searches_per_task(vfs_root, run_id),
        elapsed_seconds=round(time.monotonic() - started, 1),
    )
    return record


def score_components(routers, redact) -> dict:
    """SC-002, SC-004 and SC-005 on the labeled fixtures, with the routers the app gives each
    component: the standard tier evaluates results and runs the supervisor, the reasoning tier
    judges claims. A failing component is recorded, redacted, without stopping the others."""
    from intelligence_collector.observability.supervisor_benchmark import (
        load_fixture,
        score_claims,
        score_gap_analysis,
        score_result_evaluation,
    )

    plan = (
        ("result_evaluation", score_result_evaluation, routers.standard, "results.json"),
        ("gap_analysis", score_gap_analysis, routers.standard, "gaps.json"),
        ("claim_judge", score_claims, routers.reasoning, "claims.json"),
    )
    scores = {}
    for name, score, router, fixture in plan:
        print(f"Scoring {name} …", flush=True)
        try:
            scores[name] = score(router, load_fixture(fixture, BENCH_DIR))
        except Exception as exc:  # recorded, never fatal
            scores[name] = {"error": f"{type(exc).__name__}: {redact(exc)}"}
    return scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-live", action="store_true", help="authorize paid API calls")
    parser.add_argument("--baseline", action="store_true", help="write baseline.json")
    parser.add_argument("--only", help="comma-separated question ids to run")
    args = parser.parse_args()
    target = BENCH_DIR / ("baseline.json" if args.baseline else "benchmark_results.json")
    if not args.run_live:
        parser.error("Pass --run-live to authorize paid model, search and embedding calls")
    sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    required = ("INTELLIGENCE_REASONING_MODEL", "TAVILY_API_KEY", "TAVILY_MCP_URL")
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        print(f"Not configured: {', '.join(missing)}. Writing {target.name} with live=false.")
        target.write_text(
            json.dumps({"live": False, "missing": missing}, indent=2) + "\n", encoding="utf-8"
        )
        return 1

    # Every store of the benchmark lives under one scratch folder, never the app's own stores.
    # A short root keeps VFS paths (run id + 64-char operation hashes) under Windows' legacy
    # 260-character path limit.
    stamp = datetime.now(timezone.utc).strftime("%H%M%S")
    scratch = ROOT / ".tmp" / "b" / stamp
    os.environ.update(
        INTELLIGENCE_VFS_ROOT=str(scratch / "vfs"),
        INTELLIGENCE_CHECKPOINT_PATH=str(scratch / "checkpoints.sqlite"),
        INTELLIGENCE_RUN_REGISTRY_PATH=str(scratch / "run_registry.sqlite3"),
        INTELLIGENCE_CHROMA_PATH=str(scratch / "chroma"),
        INTELLIGENCE_OUTPUT_DIR=str(scratch / "exports"),
    )
    from intelligence_collector.models.redaction import redact_text, redact_value
    from intelligence_collector.observability.supervisor_benchmark import compare
    from intelligence_collector.observability.tracing import RunTracer
    from intelligence_collector.runtime import (
        build_model_routers,
        build_runtime,
        build_scope_agent,
    )

    routers = build_model_routers()
    runtime = build_runtime(
        routers,
        build_scope_agent(routers),
        RunTracer(),
        lifecycle=lambda _node, _status: None,
        warn=lambda message: print("WARNING", message, flush=True),
    )
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    records = []
    for question in questions:
        print(f"Running {question['id']} …", flush=True)
        record = run_question(runtime.graph, Path(runtime.vfs.root), question, redact_text)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        records.append(record)
    summary = summarize(records)
    result = {
        "live": True,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "reasoning_model": os.getenv("INTELLIGENCE_REASONING_MODEL", ""),
        "standard_model": os.getenv("INTELLIGENCE_STANDARD_MODEL", ""),
        "summary": summary,
        "runs": records,
    }
    if not args.baseline:
        baseline_path = BENCH_DIR / "baseline.json"
        baseline = (
            json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
        )
        result["components"] = score_components(routers, redact_text)
        result["compared_with_baseline"] = compare(
            summary, baseline if (baseline or {}).get("live") else None
        )
    write_result(target, result, redact_value)
    print(f"Wrote {target.relative_to(ROOT)}; artifacts in {scratch.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
