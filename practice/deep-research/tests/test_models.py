"""Structured output per provider: Groq uses its structured-output API, not a forced tool call."""

from pathlib import Path

import pytest
from dotenv import dotenv_values
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from deep_research.model_spec import ModelSpec
from deep_research.schemas.safety_verdict import SafetyVerdict
from deep_research.schemas.think import Think
from deep_research.tier import Tier
from deep_research.utils.models import _structured


def test_groq_structured_output_uses_a_strict_json_schema_not_a_tool():
    """Verify Groq schema output uses strict JSON configuration rather than a forced tool call."""
    runnable = _structured(ChatGroq(model="openai/gpt-oss-20b", api_key="test-key"), SafetyVerdict)
    bound = runnable.first.kwargs  # nothing is sent
    assert "tools" not in bound
    assert bound["response_format"]["type"] == "json_schema"
    assert bound["response_format"]["json_schema"]["strict"] is True


@pytest.mark.parametrize("tier_name,effort", [("REASONING", "medium"), ("STANDARD", "low")])
def test_example_config_preserves_reasoning_sandwich_for_primary_and_fallback(monkeypatch, tier_name, effort):
    """Check example settings construct tool-compatible models without sending requests."""
    example = Path(__file__).resolve().parents[1] / ".env.example"
    for name, value in dotenv_values(example).items():
        monkeypatch.setenv(name, value or "")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    tier = Tier(ModelSpec.from_env(tier_name, timeout=60))
    assert tier.primary.model_name == "gpt-5.4-nano"
    assert tier.primary.reasoning_effort == effort
    assert tier.primary.use_responses_api is True
    assert tier.fallback.model_name == "openai/gpt-oss-120b"
    assert tier.fallback.reasoning_effort == effort

    tools = tier.tools([Think])
    payload = tier.primary._get_request_payload([HumanMessage("Review the evidence.")], **tools.runnable.kwargs)
    assert payload["reasoning"]["effort"] == effort
    assert payload["tools"][0]["name"] == "Think"
    assert tools.fallbacks[0].kwargs["tools"][0]["function"]["name"] == "Think"


def test_fallback_options_are_optional_for_existing_environment_config(monkeypatch):
    """Keep older tier configuration compatible when fallback options are absent."""
    monkeypatch.setenv("INTELLIGENCE_REASONING_MODEL", "fake")
    monkeypatch.delenv("INTELLIGENCE_REASONING_FALLBACK_KWARGS", raising=False)
    assert ModelSpec.from_env("REASONING", timeout=60).fallback_kwargs == {}
