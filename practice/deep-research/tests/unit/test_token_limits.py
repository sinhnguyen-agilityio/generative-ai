import pytest
from pydantic import BaseModel

from intelligence_collector.models.token_limits import capped_model, output_limits_from_env


@pytest.fixture(autouse=True)
def _isolated_cap_settings(monkeypatch):
    # Other tests load the developer's .env into os.environ; these tests own these variables.
    monkeypatch.delenv("INTELLIGENCE_DISABLE_OUTPUT_CAPS", raising=False)
    monkeypatch.delenv("INTELLIGENCE_NODE_OUTPUT_TOKENS", raising=False)


class Provider(BaseModel):
    max_tokens: int | None = None


class GoogleProvider(BaseModel):
    max_output_tokens: int | None = None


def test_output_caps_can_be_disabled(monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_DISABLE_OUTPUT_CAPS", "1")
    assert output_limits_from_env() == {}
    model = Provider()
    assert capped_model(model, "verify", {}) is model


@pytest.mark.parametrize(
    "provider,field", [(Provider(), "max_tokens"), (GoogleProvider(), "max_output_tokens")]
)
def test_role_cap_clones_provider_without_mutating_shared_model(provider, field):
    limited = capped_model(provider, "verify", {"verify": 1200})
    assert getattr(limited, field) == 1200
    assert getattr(provider, field) is None


def test_lower_existing_cap_is_preserved():
    assert capped_model(Provider(max_tokens=100), "verify", {"verify": 1200}).max_tokens == 100


@pytest.mark.parametrize(
    "value", ["[]", '{"verify":true}', '{"verify":0}', '{"unknown":100}', '{"verify":1.5}']
)
def test_invalid_limits_rejected(monkeypatch, value):
    monkeypatch.setenv("INTELLIGENCE_NODE_OUTPUT_TOKENS", value)
    with pytest.raises(ValueError):
        output_limits_from_env()


def test_role_override_preserves_other_caps(monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_NODE_OUTPUT_TOKENS", '{"verify":4096}')
    limits = output_limits_from_env()
    assert limits["verify"] == 4096
    assert limits["guardrail"] == 128


def test_router_caps_primary_and_fallback_before_binding_schema():
    from intelligence_collector.contracts import ScopeAssessment
    from intelligence_collector.models.router import ModelRouter

    seen = []

    class Model(Provider):
        fail: bool = False

        def with_structured_output(self, schema):
            seen.append(self.max_tokens)
            return self

        def invoke(self, messages):
            if self.fail:
                raise RuntimeError("primary failed")
            return {"need_clarification": False, "question": "", "verification": "Scope is clear"}

    ModelRouter(Model(fail=True), Model(), output_limits={"scope": 128}).invoke_messages(
        "scope", [], ScopeAssessment
    )
    assert seen == [128, 128]


def test_retired_roles_in_an_older_env_are_ignored(monkeypatch):
    from intelligence_collector.models.token_limits import output_limits_from_env

    monkeypatch.setenv(
        "INTELLIGENCE_NODE_OUTPUT_TOKENS",
        '{"research_compressor": 1536, "research_verify": 2048, "searcher_report": 900}',
    )
    limits = output_limits_from_env()
    assert limits["searcher_report"] == 900
    assert "research_compressor" not in limits and "research_verify" not in limits
