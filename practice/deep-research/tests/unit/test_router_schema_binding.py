from langchain_groq import ChatGroq

from intelligence_collector.guardrails import ContentSafetyVerdict
from intelligence_collector.models.router import ModelRouter, _bind_schema


def test_groq_structured_output_uses_json_schema_not_tool_calling():
    """gpt-oss on Groq sometimes calls a tool named "json" instead of the bound schema tool,
    which Groq rejects with 400 tool_use_failed. Binding via response_format sends no tools."""
    model = ChatGroq(model="openai/gpt-oss-20b", api_key="unused-offline")

    bound = _bind_schema(model, ContentSafetyVerdict)

    kwargs = bound.first.kwargs
    assert "tools" not in kwargs
    assert kwargs["response_format"]["type"] == "json_schema"


def test_groq_structured_output_uses_strict_guided_decoding():
    """Without strict=True, gpt-oss's output is only best-effort schema-compliant, so a complex
    schema can make it emit malformed/empty JSON that Groq rejects post-hoc (400
    json_validate_failed with an empty failed_generation). strict=True switches Groq to
    constrained decoding, which the docs say guarantees schema-valid output -- gpt-oss-20b/120b
    are the two models that support it (Groq ignores it for other models)."""
    model = ChatGroq(model="openai/gpt-oss-20b", api_key="unused-offline")

    bound = _bind_schema(model, ContentSafetyVerdict)

    schema = bound.first.kwargs["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False
    assert set(schema["schema"]["required"]) >= set(schema["schema"]["properties"])


def test_when_primary_and_fallback_both_fail_the_primary_reason_is_reported():
    """`with_fallbacks` re-raises the primary's error, which is the root cause worth seeing."""
    import pytest

    class Failing:
        def __init__(self, name, message):
            self.model_name, self.message = name, message

        def invoke(self, messages):
            raise RuntimeError(self.message)

    router = ModelRouter(
        Failing("gemini-primary", "429 RESOURCE_EXHAUSTED"),
        Failing("groq-fallback", "413 Request too large"),
    )

    with pytest.raises(ValueError) as info:
        router.invoke_messages("synthesizer", [], ContentSafetyVerdict)

    message = str(info.value)
    assert "429 RESOURCE_EXHAUSTED" in message


def test_non_groq_models_keep_the_default_binding():
    class Fake:
        def with_structured_output(self, schema, **kwargs):
            return ("bound", schema, kwargs)

    assert _bind_schema(Fake(), ContentSafetyVerdict) == ("bound", ContentSafetyVerdict, {})
