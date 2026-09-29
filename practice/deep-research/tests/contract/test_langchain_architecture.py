from langchain_core.messages import HumanMessage, SystemMessage
from intelligence_collector.agents.prompts import ROLE_TEMPLATES, SCOPE_TEMPLATE
from intelligence_collector.models.factory import ModelConfig, ModelFactory

def test_role_prompts_are_chat_templates_and_scope_has_history_placeholder():
    messages = SCOPE_TEMPLATE.format_messages(
        conversation=[HumanMessage(content="prior"), HumanMessage(content="compare databases")],
        date="2026-09-25",
    )
    assert isinstance(messages[0], SystemMessage)
    assert any("prior" in str(message.content) for message in messages)
    assert set(ROLE_TEMPLATES) == {"planner", "supervisor", "worker", "synthesizer", "verifier", "fixer"}

def test_model_factory_uses_langchain_initializer(monkeypatch):
    calls = []
    monkeypatch.setattr("intelligence_collector.models.factory.init_chat_model", lambda *args, **kwargs: calls.append((args, kwargs)) or "model")
    model = ModelFactory().create(ModelConfig("test-model", provider="fake", timeout=4, kwargs={"temperature": 0}))
    assert model == "model" and calls[0][0] == ("test-model",) and calls[0][1]["timeout"] == 4
