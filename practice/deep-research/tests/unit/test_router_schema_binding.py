from langchain_groq import ChatGroq

from intelligence_collector.guardrails import ContentSafetyVerdict
from intelligence_collector.models.router import ModelRouter


def test_groq_structured_output_uses_json_schema_not_tool_calling():
    """gpt-oss on Groq sometimes calls a tool named "json" instead of the bound schema tool,
    which Groq rejects with 400 tool_use_failed. Binding via response_format sends no tools."""
    model = ChatGroq(model="openai/gpt-oss-20b", api_key="unused-offline")

    bound = ModelRouter._bind_schema(model, ContentSafetyVerdict)

    kwargs = bound.first.kwargs
    assert "tools" not in kwargs
    assert kwargs["response_format"]["type"] == "json_schema"


def test_when_primary_and_fallback_both_fail_both_reasons_are_reported():
    """Previously only the fallback's error surfaced, hiding why the fallback was needed."""
    import pytest

    class Failing:
        def __init__(self, name, message):
            self.model_name, self.message = name, message

        def invoke(self, messages):
            raise RuntimeError(self.message)

    router = ModelRouter(
        Failing("gemini-primary", "429 RESOURCE_EXHAUSTED"),
        Failing("groq-fallback", "413 Request too large"),
        mode="live",
    )

    with pytest.raises(ValueError) as info:
        router.invoke_messages("synthesizer", [], ContentSafetyVerdict, prompt_version="v")

    message = str(info.value)
    assert "gemini-primary failed: 429 RESOURCE_EXHAUSTED" in message
    assert "groq-fallback failed: 413 Request too large" in message


def test_non_groq_models_keep_the_default_binding():
    class Fake:
        def with_structured_output(self, schema, **kwargs):
            return ("bound", schema, kwargs)

    assert ModelRouter._bind_schema(Fake(), ContentSafetyVerdict) == ("bound", ContentSafetyVerdict, {})
