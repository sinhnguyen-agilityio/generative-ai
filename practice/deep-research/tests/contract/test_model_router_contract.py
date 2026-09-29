import json

from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from intelligence_collector.models.router import ModelRouter
def test_model_router_falls_back_once():
    def primary(role, prompt): raise RuntimeError("timeout")
    def fallback(role, prompt): return {"ok": True}
    router=ModelRouter(primary, fallback)
    result = router.invoke_messages("scope", [HumanMessage(content="hello")], _Decision, prompt_version="t.v1")
    assert result == _Decision(ok=True)


class _Decision(BaseModel):
    ok: bool


def test_invoke_messages_parses_multi_block_thinking_content():
    """Reasoning-tier models (e.g. Gemini with reasoning_effort set) can return `.content` as a
    list of blocks (a thinking block plus a text block) rather than a plain string. The final
    JSON answer must still parse instead of raising "model output is not structured"."""

    class ThinkingModel:
        def invoke(self, messages):
            return AIMessage(content=[
                {"type": "thinking", "thinking": "reasoning about the answer..."},
                {"type": "text", "text": json.dumps({"ok": True})},
            ])

    router = ModelRouter(ThinkingModel())

    result = router.invoke_messages(
        "scope", [HumanMessage(content="hi")], _Decision, prompt_version="test.v1",
    )

    assert result == _Decision(ok=True)


def test_invoke_messages_uses_provider_native_structured_output():
    """Models exposing with_structured_output must be bound to the schema rather than trusted to
    emit bare JSON text (Gemini tends to wrap it in Markdown fences)."""
    bound = []

    class StructuredModel:
        model_name = "native"

        def with_structured_output(self, schema):
            bound.append(schema)
            return self

        def invoke(self, messages):
            return _Decision(ok=True)

    router = ModelRouter(StructuredModel())

    result = router.invoke_messages(
        "research_compressor", [HumanMessage(content="hi")], _Decision, prompt_version="t.v1",
    )

    assert result == _Decision(ok=True)
    assert bound == [_Decision]


def test_router_requires_a_primary_model():
    import pytest

    with pytest.raises(RuntimeError, match="primary model"):
        ModelRouter(None)
