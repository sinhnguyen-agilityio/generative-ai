import pytest

from intelligence_collector.config import RoleModelSettings
from intelligence_collector.models.role_router import RoleModelRouter


class RecordingRouter:
    def __init__(self, name):
        self.name = name
        self.roles = []

    def invoke(self, role, prompt):
        self.roles.append(role)
        return self.name

    def invoke_with_tools(self, role, messages, tools, **kwargs):
        self.roles.append(role)
        return self.name


def test_scope_planning_and_verification_use_deep_reasoning_profile():
    deep = RecordingRouter("deep")
    standard = RecordingRouter("standard")
    router = RoleModelRouter(deep, standard)

    assert router.invoke("scope", "clarify") == "deep"
    assert router.invoke("planner", "plan") == "deep"
    assert router.invoke("verify", "verify") == "deep"
    assert deep.roles == ["scope", "planner", "verify"]


def test_implementation_roles_use_standard_profile():
    deep = RecordingRouter("deep")
    standard = RecordingRouter("standard")
    router = RoleModelRouter(deep, standard)

    for role in ("researcher", "research_compressor", "synthesizer", "fixer"):
        assert router.invoke(role, role) == "standard"
    assert not deep.roles


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


def test_researcher_tool_calls_use_standard_profile():
    deep = RecordingRouter("deep")
    standard = RecordingRouter("standard")
    router = RoleModelRouter(deep, standard)

    assert router.invoke_with_tools("researcher", [], []) == "standard"
    assert standard.roles == ["researcher"]
    assert not deep.roles
