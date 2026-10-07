from intelligence_collector.config import RoleModelSettings
from intelligence_collector.models.factory import ModelFactory


def test_live_routers_default_to_one_explicit_attempt(monkeypatch):
    """Live execution is cost-bounded by default; retries require explicit configuration."""
    monkeypatch.setattr(
        "intelligence_collector.models.factory.init_chat_model", lambda model, **kwargs: object()
    )
    settings = RoleModelSettings(reasoning_model="r-model", standard_model="s-model")
    monkeypatch.delenv("INTELLIGENCE_MODEL_MAX_ATTEMPTS", raising=False)

    routers = ModelFactory().create_routers(settings)

    assert routers.reasoning.max_attempts == 1
    assert routers.standard.max_attempts == 1


def test_each_tier_uses_its_own_configurable_timeout_for_primary_and_fallback(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "intelligence_collector.models.factory.init_chat_model",
        lambda model, **kwargs: seen.append((model, kwargs["timeout"])) or object(),
    )
    settings = RoleModelSettings(
        reasoning_model="r-model",
        standard_model="s-model",
        reasoning_fallback_model="r-fallback",
        reasoning_timeout=150.0,
        standard_timeout=45.0,
    )

    ModelFactory().create_routers(settings)

    assert ("r-model", 150.0) in seen and ("r-fallback", 150.0) in seen
    assert ("s-model", 45.0) in seen


def test_timeouts_are_read_from_env_and_bounded(monkeypatch):
    import pytest

    monkeypatch.setenv("INTELLIGENCE_REASONING_MODEL", "r")
    monkeypatch.setenv("INTELLIGENCE_STANDARD_MODEL", "s")
    monkeypatch.setenv("INTELLIGENCE_REASONING_TIMEOUT", "200")
    monkeypatch.delenv("INTELLIGENCE_STANDARD_TIMEOUT", raising=False)

    settings = RoleModelSettings.from_env()

    assert settings.reasoning_timeout == 200.0
    assert settings.standard_timeout == 60.0  # default

    monkeypatch.setenv("INTELLIGENCE_REASONING_TIMEOUT", "1")
    with pytest.raises(ValueError, match="INTELLIGENCE_REASONING_TIMEOUT"):
        RoleModelSettings.from_env()


def test_openai_reasoning_effort_with_tools_uses_the_responses_api(monkeypatch):
    """gpt-5.x rejects function tools + reasoning_effort on /v1/chat/completions."""
    from langchain_core.messages import HumanMessage

    from intelligence_collector.models.factory import ModelConfig

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    tool = {"type": "function", "function": {"name": "write_todos", "description": "d",
                                             "parameters": {"type": "object", "properties": {}}}}
    factory = ModelFactory()

    model = factory.create(
        ModelConfig("gpt-5.4-nano", provider="openai", kwargs={"reasoning_effort": "low"})
    )
    bound = model.bind_tools([tool])
    payload = bound.bound._get_request_payload([HumanMessage("hi")], **bound.kwargs)
    assert model.use_responses_api is True
    assert payload["reasoning"] == {"effort": "low"} and "reasoning_effort" not in payload

    # An explicit choice is kept; without reasoning_effort nothing changes.
    explicit = factory.create(ModelConfig(
        "gpt-5.4-nano", provider="openai",
        kwargs={"reasoning_effort": "low", "use_responses_api": False},
    ))
    plain = factory.create(ModelConfig("gpt-4.1-mini", provider="openai"))
    assert explicit.use_responses_api is False and plain.use_responses_api is None
