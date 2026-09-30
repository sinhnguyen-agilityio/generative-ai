from intelligence_collector.config import RoleModelSettings
from intelligence_collector.models.factory import ModelFactory
from intelligence_collector.models.router import MAX_MODEL_ATTEMPTS


def test_reasoning_router_gets_every_retry_attempt_the_router_allows(monkeypatch):
    """Groq's strict structured-output decoding for gpt-oss is intermittently flaky server-side
    (json_validate_failed on a fraction of otherwise-identical requests) -- retrying is the only
    lever available for that, so the reasoning tier (which carries the report-judge "verify"
    call) should get the router's maximum retry budget, not the default."""
    monkeypatch.setattr(
        "intelligence_collector.models.factory.init_chat_model", lambda model, **kwargs: object()
    )
    settings = RoleModelSettings(reasoning_model="r-model", standard_model="s-model")

    routers = ModelFactory().create_routers(settings)

    assert routers.reasoning.max_attempts == MAX_MODEL_ATTEMPTS
    assert routers.standard.max_attempts == 2


def test_each_tier_uses_its_own_configurable_timeout_for_primary_and_fallback(monkeypatch):
    seen = []
    monkeypatch.setattr(
        "intelligence_collector.models.factory.init_chat_model",
        lambda model, **kwargs: seen.append((model, kwargs["timeout"])) or object(),
    )
    settings = RoleModelSettings(
        reasoning_model="r-model", standard_model="s-model",
        reasoning_fallback_model="r-fallback", reasoning_timeout=150.0, standard_timeout=45.0,
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
