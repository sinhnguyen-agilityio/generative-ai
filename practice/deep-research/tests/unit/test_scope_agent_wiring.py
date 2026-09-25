"""Role-tier model wiring for the live app (app._get_role_router / _get_scope_agent).

Every agent (Scope, Planner, Researcher, Report Writer, Acceptance Verifier, Fixer) shares one
``RoleModelRouter`` instance by default (see build.py/nodes.py/app.py), which dispatches each
call to the reasoning profile (scope, planner, verify) or the standard profile (researcher,
research_compressor, synthesizer, fixer) based on the role name passed to ``invoke_messages``.
"""

from types import SimpleNamespace

from intelligence_collector.app import _get_role_router, _get_scope_agent
from intelligence_collector.models.role_router import RoleModelRouter


class _SessionState(dict):
    """Minimal stand-in for Streamlit's session_state: dict membership + attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self[name] = value


def _fake_st():
    return SimpleNamespace(session_state=_SessionState())


def _fake_tracer():
    return SimpleNamespace(model_event=lambda **_: None, callbacks=[])


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

    router = _get_role_router(_fake_st(), _fake_tracer())

    assert isinstance(router, RoleModelRouter)
    assert calls == [
        ("gpt-5.4", "openai"),
        ("gemini-2.5-flash", "google_genai"),
        ("gpt-5.4", "openai"),  # standard tier's fallback
    ]
    assert router.deep_reasoning.primary.model_name == "gpt-5.4"
    assert router.deep_reasoning.fallback is None
    assert router.standard.primary.model_name == "gemini-2.5-flash"
    assert router.standard.fallback.model_name == "gpt-5.4"
    assert router.deep_reasoning.mode == router.standard.mode == "live"


def test_no_fallback_configured_means_no_fallback_on_either_tier(monkeypatch):
    _set_role_env(monkeypatch)
    _recording_init_chat_model(monkeypatch)

    router = _get_role_router(_fake_st(), _fake_tracer())

    assert router.deep_reasoning.fallback is None
    assert router.standard.fallback is None


def test_missing_reasoning_model_returns_none(monkeypatch):
    monkeypatch.delenv("INTELLIGENCE_REASONING_MODEL", raising=False)

    assert _get_role_router(_fake_st(), _fake_tracer()) is None
    assert _get_scope_agent(_fake_st(), _fake_tracer()) is None


def test_role_router_is_cached_on_session_state(monkeypatch):
    _set_role_env(monkeypatch)
    calls = _recording_init_chat_model(monkeypatch)
    st = _fake_st()

    first = _get_role_router(st, _fake_tracer())
    second = _get_role_router(st, _fake_tracer())

    assert first is second
    assert len(calls) == 2  # reasoning + standard primaries only; no new construction on reuse


def test_scope_agent_shares_the_role_router_and_is_cached(monkeypatch):
    _set_role_env(monkeypatch)
    _recording_init_chat_model(monkeypatch)
    st = _fake_st()

    agent = _get_scope_agent(st, _fake_tracer())
    second = _get_scope_agent(st, _fake_tracer())

    assert agent is second
    assert agent.router is st.session_state.role_router
