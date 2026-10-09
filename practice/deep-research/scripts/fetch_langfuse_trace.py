"""Fetch the latest project trace or a research thread's trace using dotenv credentials.

Examples:
    python scripts/fetch_langfuse_trace.py
    python scripts/fetch_langfuse_trace.py --trace-id <trace-id>
    python scripts/fetch_langfuse_trace.py --thread-id <research-thread-id> --hours 168

Requires Langfuse SDK 4.8.1+ and a Cloud/v4 server. Only reads Langfuse.
Full inputs/outputs stay in the ignored .data/traces directory.
The time window bounds extraction; a running or older thread may require another fetch.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError

ROOT = Path(__file__).resolve().parents[1]


def fetch_pages(fetch, **filters) -> list:
    """Read all cursor pages without silently truncating observations or scores."""
    rows, seen = {}, set()
    cursor = None
    while True:
        page = fetch(limit=100, cursor=cursor, **filters)
        rows.update((row.id, row) for row in page.data)
        cursor = page.meta.cursor
        if not cursor:
            return list(rows.values())
        if cursor in seen:
            raise RuntimeError("Langfuse returned a repeated pagination cursor")
        seen.add(cursor)


def summarize(observations: list[dict], scores: list[dict]) -> dict:
    """Report measured generation usage and span timing without dumping research content.

    Nested span durations are not summed: parallel and parent spans overlap.
    Uninstrumented embedding, search, or evaluation calls are outside these totals.
    """
    generations = [row for row in observations if row["type"] == "GENERATION"]
    starts = [datetime.fromisoformat(row["startTime"]) for row in observations]
    ends = [datetime.fromisoformat(row["endTime"]) for row in observations if row.get("endTime")]
    costs = [row["totalCost"] for row in generations if row.get("totalCost") is not None]
    usage = Counter()
    tool_calls = Counter()
    state_errors = []
    for row in observations:
        output = row.get("output")
        if isinstance(output, str):
            try:
                output = json.loads(output)
            except ValueError:
                continue
        if not isinstance(output, dict):
            continue
        if row["type"] == "GENERATION":
            tool_calls.update(call["name"] for call in output.get("tool_calls", []))
        for key in ("verification", "evaluation"):
            value = output.get(key)
            if isinstance(value, dict) and value.get("error"):
                state_errors.append({"observation_id": row["id"], "name": row.get("name"), "state": key, "error": value["error"]})
    for row in generations:
        usage.update(row.get("usageDetails") or {})
    return {
        "observations": len(observations),
        "generations": len(generations),
        "models": dict(Counter(row.get("model") or "unknown" for row in generations)),
        "usage_details": dict(usage),
        "tool_calls": dict(tool_calls),
        "reported_generation_cost_usd": sum(costs) if costs else None,
        "generations_with_cost": len(costs),
        "first_start": min(starts).isoformat() if starts else None,
        "last_end": max(ends).isoformat() if ends else None,
        "observed_elapsed_seconds": (max(ends) - min(starts)).total_seconds() if starts and ends else None,
        "open_observations": sum(not row.get("endTime") for row in observations),
        "errors": [{"id": row["id"], "name": row.get("name")} for row in observations if row.get("level") == "ERROR"],
        "state_errors": state_errors,
        "scores": [{"name": row.get("name"), "value": row.get("value"), "observation_id": row.get("observationId")} for row in scores],
    }


def main() -> None:
    """Load .env, select a trace, save a bounded snapshot, and print a compact summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--trace-id", help="Fetch a specific trace")
    selection.add_argument("--thread-id", help="Derive the same trace ID as the research runner")
    parser.add_argument("--name", help="Filter root observation names when selecting the latest trace")
    parser.add_argument("--hours", type=float, default=24, help="Lookback window in hours (default: 24)")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    if not 0 < args.hours <= 24 * 365:
        parser.error("--hours must be positive and at most one year")
    load_dotenv(args.env_file, override=False)
    missing = [name for name in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY") if not os.getenv(name)]
    if missing:
        parser.error("Missing configuration: " + ", ".join(missing))

    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=args.hours)
    client = Langfuse(tracing_enabled=False, timeout=30)
    try:
        trace_id = args.trace_id or (Langfuse.create_trace_id(seed=args.thread_id) if args.thread_id else None)
        if not trace_id:
            roots = fetch_pages(client.api.observations.get_many, fields="core,basic", is_root_observation=True, name=args.name, from_start_time=start, to_start_time=end)
            roots = [row for row in roots if row.trace_id]
            if not roots:
                parser.exit(1, "No trace roots found in this window; increase --hours or pass --trace-id.\n")
            trace_id = max(roots, key=lambda row: row.start_time).trace_id
        if len(trace_id) != 32 or any(c not in "0123456789abcdef" for c in trace_id):
            parser.error("Trace ID must be 32 lowercase hexadecimal characters")
        observations = fetch_pages(
            client.api.observations.get_many, trace_id=trace_id,
            fields="core,basic,time,io,model,usage,metrics,trace_context",
            from_start_time=start, to_start_time=end,
        )
        if not observations:
            parser.exit(1, "No observations found for this trace in the selected window.\n")
        scores = fetch_pages(client.api.scores_v3.get_many_v3, trace_id=trace_id, from_timestamp=start, to_timestamp=end)
        data = sorted((row.model_dump(mode="json") for row in observations), key=lambda row: row["startTime"])
        score_data = [row.model_dump(mode="json") for row in scores]
        summary = summarize(data, score_data)
        project_id = data[0]["projectId"]
        base_url = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
        snapshot = {
            "trace_id": trace_id,
            "trace_url": f"{base_url.rstrip('/')}/project/{project_id}/traces/{trace_id}",
            "fetched_at": end.isoformat(),
            "window": {"from": start.isoformat(), "to": end.isoformat()},
            "summary": summary, "observations": data, "scores": score_data,
        }
        destination = ROOT / ".data" / "traces" / f"{trace_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"trace_id": trace_id, "trace_url": snapshot["trace_url"], "saved_to": str(destination), **summary}, ensure_ascii=False, indent=2))
    except ApiError as exc:
        parser.exit(1, f"Langfuse request failed (HTTP {exc.status_code}); check project credentials and API access.\n")
    finally:
        client.shutdown()


if __name__ == "__main__":
    main()
