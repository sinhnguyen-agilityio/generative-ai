import pytest

from intelligence_collector.config import RoleModelSettings


def test_role_model_settings_load_provider_specific_kwargs(monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_REASONING_MODEL", "reasoning-model")
    monkeypatch.setenv("INTELLIGENCE_STANDARD_MODEL", "standard-model")
    monkeypatch.setenv("INTELLIGENCE_REASONING_KWARGS", '{"reasoning_effort":"high"}')
    settings = RoleModelSettings.from_env()
    assert settings.reasoning_kwargs == {"reasoning_effort": "high"}


def test_role_model_settings_require_both_profiles(monkeypatch):
    monkeypatch.delenv("INTELLIGENCE_REASONING_MODEL", raising=False)
    monkeypatch.delenv("INTELLIGENCE_STANDARD_MODEL", raising=False)
    with pytest.raises(ValueError, match="are required"):
        RoleModelSettings.from_env()


def test_role_model_settings_fallbacks_default_to_none(monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_REASONING_MODEL", "reasoning-model")
    monkeypatch.setenv("INTELLIGENCE_STANDARD_MODEL", "standard-model")
    for name in (
        "INTELLIGENCE_REASONING_FALLBACK_MODEL",
        "INTELLIGENCE_STANDARD_FALLBACK_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    settings = RoleModelSettings.from_env()

    assert settings.reasoning_fallback_model is None
    assert settings.standard_fallback_model is None
