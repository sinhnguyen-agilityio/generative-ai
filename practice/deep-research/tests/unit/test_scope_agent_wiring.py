"""Model-tier wiring for the live app (ui.resources.get_model_routers / get_scope_agent).

The app builds a reasoning-tier and a standard-tier router (each with its own fallback) and
hands each agent the tier it needs: Scope/Planner/Report Judge/Feedback classifier get the
reasoning tier; Researcher/Report Writer (and the scope guardrail) get the standard tier.
"""

from types import SimpleNamespace

import pytest

from intelligence_collector.ui import resources
from intelligence_collector.models.factory import ModelRouters


class _SessionState(dict):
    """Minimal stand-in for Streamlit's session_state: dict membership + attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


@pytest.fixture(autouse=True)
def st(monkeypatch):
    """A fresh fake Streamlit module for each test: session-scoped caching starts empty."""
    fake = SimpleNamespace(session_state=_SessionState())
    monkeypatch.setattr(resources, "st", fake)
    return fake


def _recording_init_chat_model(monkeypatch):
    calls = []

    def fake(model, **kwargs):
        calls.append((model, kwargs.get("model_provider")))
        return SimpleNamespace(model_name=model, provider=kwargs.get("model_provider"))

    monkeypatch.setattr("intelligence_collector.models.factory.init_chat_model", fake)
    return calls


def _set_role_env(monkeypatch, *, reasoning_fallback=False, standard_fallback=False):
    monkeypatch.setenv("INTELLIGENCE_REASONING_MODEL", "gpt-5.4")
    monkeypatch.setenv("INTELLIGENCE_REASONING_PROVIDER", "openai")
    monkeypatch.setenv("INTELLIGENCE_STANDARD_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("INTELLIGENCE_STANDARD_PROVIDER", "google_genai")
    if reasoning_fallback:
        monkeypatch.setenv("INTELLIGENCE_REASONING_FALLBACK_MODEL", "gpt-5.3")
        monkeypatch.setenv("INTELLIGENCE_REASONING_FALLBACK_PROVIDER", "openai")
    else:
        monkeypatch.delenv("INTELLIGENCE_REASONING_FALLBACK_MODEL", raising=False)
        monkeypatch.delenv("INTELLIGENCE_REASONING_FALLBACK_PROVIDER", raising=False)
    if standard_fallback:
        monkeypatch.setenv("INTELLIGENCE_STANDARD_FALLBACK_MODEL", "gpt-5.4")
        monkeypatch.setenv("INTELLIGENCE_STANDARD_FALLBACK_PROVIDER", "openai")
    else:
        monkeypatch.delenv("INTELLIGENCE_STANDARD_FALLBACK_MODEL", raising=False)
        monkeypatch.delenv("INTELLIGENCE_STANDARD_FALLBACK_PROVIDER", raising=False)


def test_reasoning_tier_is_gpt_and_standard_tier_is_gemini(monkeypatch):
    _set_role_env(monkeypatch, standard_fallback=True)
    calls = _recording_init_chat_model(monkeypatch)

    router = resources.get_model_routers()

    assert isinstance(router, ModelRouters)
    assert calls == [
        ("gpt-5.4", "openai"),
        ("gemini-2.5-flash", "google_genai"),
        ("gpt-5.4", "openai"),  # standard tier's fallback
    ]
    assert router.reasoning.primary.model_name == "gpt-5.4"
    assert router.reasoning.fallback is None
    assert router.standard.primary.model_name == "gemini-2.5-flash"
    assert router.standard.fallback.model_name == "gpt-5.4"


def test_no_fallback_configured_means_no_fallback_on_either_tier(monkeypatch):
    _set_role_env(monkeypatch)
    _recording_init_chat_model(monkeypatch)

    router = resources.get_model_routers()

    assert router.reasoning.fallback is None
    assert router.standard.fallback is None


def test_missing_reasoning_model_returns_none(monkeypatch):
    monkeypatch.delenv("INTELLIGENCE_REASONING_MODEL", raising=False)

    assert resources.get_model_routers() is None
    assert resources.get_scope_agent() is None


def test_model_routers_are_cached_on_session_state(monkeypatch):
    _set_role_env(monkeypatch)
    calls = _recording_init_chat_model(monkeypatch)

    first = resources.get_model_routers()
    second = resources.get_model_routers()

    assert first is second
    assert len(calls) == 2  # reasoning + standard primaries only; no new construction on reuse


def test_scope_agent_gets_reasoning_router_and_standard_guardrail_and_is_cached(monkeypatch, st):
    _set_role_env(monkeypatch)
    _recording_init_chat_model(monkeypatch)

    agent = resources.get_scope_agent()
    second = resources.get_scope_agent()

    assert agent is second
    assert agent.router is st.session_state.model_routers.reasoning
    assert agent.guardrail_router is st.session_state.model_routers.standard
