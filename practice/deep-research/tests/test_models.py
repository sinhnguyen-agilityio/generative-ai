"""Structured output per provider: Groq uses its structured-output API, not a forced tool call."""

from langchain_groq import ChatGroq

from deep_research.schemas.safety_verdict import SafetyVerdict
from deep_research.utils.models import _structured


def test_groq_structured_output_uses_a_strict_json_schema_not_a_tool():
    """Verify Groq schema output uses strict JSON configuration rather than a forced tool call."""
    runnable = _structured(ChatGroq(model="openai/gpt-oss-20b", api_key="test-key"), SafetyVerdict)
    bound = runnable.first.kwargs  # nothing is sent
    assert "tools" not in bound
    assert bound["response_format"]["type"] == "json_schema"
    assert bound["response_format"]["json_schema"]["strict"] is True
