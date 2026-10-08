"""Structured output per provider: Groq uses its structured-output API, not a forced tool call."""

from langchain_groq import ChatGroq

from deep_research.models import _structured
from deep_research.state import SafetyVerdict


def test_groq_structured_output_uses_a_strict_json_schema_not_a_tool():
    runnable = _structured(ChatGroq(model="openai/gpt-oss-20b", api_key="test-key"), SafetyVerdict)
    bound = runnable.first.kwargs  # nothing is sent
    assert "tools" not in bound
    assert bound["response_format"]["type"] == "json_schema"
    assert bound["response_format"]["json_schema"]["strict"] is True
