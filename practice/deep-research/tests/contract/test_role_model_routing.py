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


def test_planning_and_verification_use_deep_reasoning_profile():
    deep = RecordingRouter("deep")
    standard = RecordingRouter("standard")
    router = RoleModelRouter(deep, standard)

    assert router.invoke("planner", "plan") == "deep"
    assert router.invoke("verifier", "verify") == "deep"
    assert deep.roles == ["planner", "verifier"]


def test_implementation_roles_use_standard_profile():
    deep = RecordingRouter("deep")
    standard = RecordingRouter("standard")
    router = RoleModelRouter(deep, standard)

    for role in ("supervisor", "worker", "synthesizer", "fixer"):
        assert router.invoke(role, role) == "standard"
    assert not deep.roles


def test_role_model_settings_load_provider_specific_kwargs(monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_REASONING_MODEL", "reasoning-model")
    monkeypatch.setenv("INTELLIGENCE_STANDARD_MODEL", "standard-model")
    monkeypatch.setenv("INTELLIGENCE_REASONING_KWARGS", '{"reasoning_effort":"xhigh"}')
    settings = RoleModelSettings.from_env()
    assert settings.reasoning_kwargs == {"reasoning_effort": "xhigh"}


def test_role_model_settings_require_both_profiles(monkeypatch):
    monkeypatch.delenv("INTELLIGENCE_REASONING_MODEL", raising=False)
    monkeypatch.delenv("INTELLIGENCE_STANDARD_MODEL", raising=False)
    with pytest.raises(ValueError, match="are required"):
        RoleModelSettings.from_env()
