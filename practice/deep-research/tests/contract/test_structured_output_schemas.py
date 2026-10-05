"""Every model-output schema must be accepted by OpenAI's strict structured outputs."""

import pytest
from openai.lib._pydantic import to_strict_json_schema

from intelligence_collector.contracts import (
    DraftAssessment,
    FeedbackDecision,
    PageSummary,
    Reflection,
    ReportJudgeVerdict,
    ResearchBrief,
    ResearchFindings,
    ResearchEvidenceVerdict,
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
    PageSummary,
    ResearchEvidenceVerdict,
    DraftAssessment,
    Reflection,
    ReportJudgeVerdict,
    FeedbackDecision,
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
