import pytest

from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.contracts import ResearchBrief, ScopeAssessment
from intelligence_collector.models.router import ModelRouter
from tests.integration.test_langchain_scope import (
    RecordingChatModel,
    _brief_payload,
    _clear_payload,
    _guardrail_payload,
)

def test_scope_llm_asks_one_focused_question_for_ambiguous_request():
    fake = RecordingChatModel([_guardrail_payload(), {
        "need_clarification": True,
        "question": "Which databases and workload? Choose two named databases for one workload.",
        "verification": "",
    }])
    result = ScopeAgent(ModelRouter(fake, fake)).assess("Compare databases")
    assert result.need_clarification is True
    assert result.question

def test_scope_llm_returns_clear_brief():
    fake = RecordingChatModel([_guardrail_payload(), _clear_payload(), _brief_payload()])
    result = ScopeAgent(ModelRouter(fake, fake)).assess("Compare PostgreSQL and SQLite")
    assert isinstance(result, ResearchBrief) and result.research_question.startswith("I want")


def test_clarification_without_a_question_is_rejected():
    invalid = {
        "need_clarification": True,
        "question": "",
        "verification": "",
    }
    fake = RecordingChatModel([_guardrail_payload(), invalid, invalid])

    with pytest.raises(ValueError, match="invalid structured output"):
        ScopeAgent(ModelRouter(fake)).assess("Compare databases")


def test_scope_assessment_schema_has_exactly_three_decision_fields():
    assert set(ScopeAssessment.model_fields) == {
        "need_clarification",
        "question",
        "verification",
    }
