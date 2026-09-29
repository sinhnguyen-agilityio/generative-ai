import pytest

from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.guardrails import GuardrailViolation
from intelligence_collector.models.router import ModelRouter
from tests.integration.test_langchain_scope import RecordingChatModel, _guardrail_payload


def test_blocked_content_never_reaches_the_scope_model():
    fake = RecordingChatModel([{"allowed": False, "reason": "Contains vulgar language."}])
    agent = ScopeAgent(ModelRouter(fake))

    with pytest.raises(GuardrailViolation, match="vulgar language"):
        agent.clarify_with_user("some blocked request")

    # Only the guardrail call happened; the expensive scope model was never invoked.
    assert len(fake.calls) == 1


def test_allowed_content_proceeds_to_the_scope_model():
    fake = RecordingChatModel([
        _guardrail_payload(),
        {
            "need_clarification": True,
            "question": "Which databases and workload?",
            "verification": "",
        },
    ])
    agent = ScopeAgent(ModelRouter(fake))

    result = agent.clarify_with_user("Compare databases")

    assert result.need_clarification is True
    assert len(fake.calls) == 2


def test_check_content_safety_uses_the_guardrail_role():
    traces = []
    fake = RecordingChatModel([_guardrail_payload()])
    router = ModelRouter(fake, trace=lambda **event: traces.append(event))
    agent = ScopeAgent(router)

    verdict = agent.check_content_safety("Compare databases")

    assert verdict.allowed is True
    assert traces[0]["role"] == "guardrail"
