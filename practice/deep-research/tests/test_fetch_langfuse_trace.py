"""Trace extraction pagination and summaries using offline SDK-shaped responses."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from langfuse.api.commons.types.observation_v2 import ObservationV2

from scripts.fetch_langfuse_trace import fetch_pages, summarize


def test_fetch_pages_reads_every_page_and_deduplicates_observations():
    """Preserve all observations when page boundaries overlap."""
    calls = []
    pages = {
        None: SimpleNamespace(data=[SimpleNamespace(id="a"), SimpleNamespace(id="b")], meta=SimpleNamespace(cursor="next")),
        "next": SimpleNamespace(data=[SimpleNamespace(id="b"), SimpleNamespace(id="c")], meta=SimpleNamespace(cursor=None)),
    }

    def fetch(**kwargs):
        """Return the page selected by the incoming cursor."""
        calls.append(kwargs)
        return pages[kwargs["cursor"]]

    assert [row.id for row in fetch_pages(fetch, trace_id="trace")] == ["a", "b", "c"]
    assert len(calls) == 2
    assert all(call["trace_id"] == "trace" for call in calls)


def test_repeated_cursor_fails_instead_of_looping_forever():
    """Stop on an invalid server pagination response."""
    def fetch(**kwargs):
        """Return a cursor that never advances."""
        return SimpleNamespace(data=[], meta=SimpleNamespace(cursor="same"))

    with pytest.raises(RuntimeError, match="repeated"):
        fetch_pages(fetch)


def test_summary_uses_sdk_json_aliases_and_detects_caught_state_errors():
    """Aggregate generation usage without counting parent span costs or hiding node errors."""
    start = datetime(2026, 10, 9, tzinfo=timezone.utc)
    generation = ObservationV2(
        id="model", trace_id="trace", project_id="project", type="GENERATION",
        start_time=start, end_time=start.replace(second=3), model="gpt-5.4-nano",
        total_cost=0.01, usage_details={"input": 10, "output": 5, "total": 15},
        output=json.dumps({"tool_calls": [{"name": "WebSearch"}]}),
    ).model_dump(mode="json")
    span = ObservationV2(
        id="parent", project_id="project", type="SPAN", name="evaluate",
        start_time=start, end_time=start.replace(second=5), total_cost=1,
        output=json.dumps({"evaluation": {"error": "503 UNAVAILABLE"}}),
    ).model_dump(mode="json")
    summary = summarize([generation, span], [])
    assert summary["reported_generation_cost_usd"] == 0.01
    assert summary["observed_elapsed_seconds"] == 5
    assert summary["usage_details"]["total"] == 15
    assert summary["tool_calls"] == {"WebSearch": 1}
    assert summary["errors"] == []
    assert summary["state_errors"][0]["error"] == "503 UNAVAILABLE"


def test_missing_usage_and_cost_are_not_reported_as_known_zero_cost():
    """An unmetered span cannot establish the model's cost."""
    row = {"id": "span", "type": "SPAN", "startTime": "2026-10-09T00:00:00Z"}
    summary = summarize([row], [])
    assert summary["reported_generation_cost_usd"] is None
    assert summary["open_observations"] == 1
