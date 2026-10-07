"""Every model-output schema must be accepted by OpenAI's strict structured outputs."""

import pytest
from openai.lib._pydantic import to_strict_json_schema

from intelligence_collector.contracts import (
    ClaimJudgeBatch,
    DraftAssessment,
    GapCoverageBatch,
    QueryRewrite,
    ResultEvaluationBatch,
    SearcherReport,
    SearcherReportDraft,
    FeedbackDecision,
    Reflection,
    ReportJudgeVerdict,
    ResearchBrief,
    ResearchFindings,
    ScopeAssessment,
    TodoPlan,
)
from intelligence_collector.guardrails import ContentSafetyVerdict

ROUTER_SCHEMAS = [
    ContentSafetyVerdict,
    ScopeAssessment,
    ResearchBrief,
    TodoPlan,
    ResearchFindings,
    DraftAssessment,
    Reflection,
    ReportJudgeVerdict,
    FeedbackDecision,
    # Supervisor–searcher workflow (spec 002)
    ResultEvaluationBatch,
    QueryRewrite,
    SearcherReportDraft,
    GapCoverageBatch,
    ClaimJudgeBatch,
]


def _free_form_objects(schema, path="$"):
    """Paths of objects without fixed properties (maps), which strict mode rejects."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" not in schema and "$ref" not in schema:
            yield path
        for key, value in schema.items():
            yield from _free_form_objects(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            yield from _free_form_objects(value, f"{path}[{i}]")


@pytest.mark.parametrize("schema", ROUTER_SCHEMAS, ids=lambda s: s.__name__)
def test_model_output_schema_has_no_free_form_maps(schema):
    assert list(_free_form_objects(to_strict_json_schema(schema))) == []


def test_stored_searcher_report_matches_the_contract_example():
    """contracts/searcher-report.md: the stored report shape."""
    report = SearcherReport.model_validate(
        {
            "todo_id": "t-1",
            "points": [
                {"point_id": "E1", "status": "answered", "lookup_used": True,
                 "support": [{"source_id": "src_3f2a9c1b7d4e", "quote": "only one writer at a time"}]},
                {"point_id": "E2", "status": "missing", "lookup_used": True, "support": []},
            ],
            "findings": [{"text": "SQLite serializes writes.", "source_ids": ["src_3f2a9c1b7d4e"]}],
            "accepted_count": 3,
            "rejected_count": 4,
            "queries": ["SQLite concurrent writes", "SQLite WAL writer lock"],
            "stop_reason": "no_progress",
            "failure_kind": None,
            "report_status": "corrected",
            "corrections": ["E2: quote not found in src_9a; downgraded to missing"],
        }
    )
    dumped = report.model_dump(mode="json")
    assert set(dumped) >= {
        "todo_id", "points", "findings", "accepted_count", "rejected_count", "queries",
        "stop_reason", "failure_kind", "report_status", "corrections",
    }
    assert "confidence" not in dumped
