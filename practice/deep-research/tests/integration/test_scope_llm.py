import pytest

from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.contracts import ResearchBrief, ScopeDecision
from intelligence_collector.models.router import ModelRouter
from tests.integration.test_langchain_scope import RecordingChatModel, _brief_payload, _clear_payload

def test_scope_llm_asks_one_focused_question_for_ambiguous_request():
    fake = RecordingChatModel([{"decision": "NEEDS_CLARIFICATION", "clarification_question": "Which databases and workload?", "missing_fields": ["research_boundaries"], "narrowing_suggestions": ["Compare two named databases for one workload."], "reason": "too broad"}])
    result = ScopeAgent(ModelRouter(fake, fake, mode="fake")).assess("Compare databases")
    assert result.decision == ScopeDecision.NEEDS_CLARIFICATION
    assert result.clarification_question
    assert result.narrowing_suggestions

def test_scope_llm_returns_clear_brief():
    fake = RecordingChatModel([_clear_payload(), _brief_payload()])
    result = ScopeAgent(ModelRouter(fake, fake, mode="fake")).assess("Compare PostgreSQL and SQLite")
    assert isinstance(result, ResearchBrief) and result.research_question.startswith("I want")


def test_broad_scope_without_narrowing_options_is_rejected():
    invalid = {
        "decision": "NEEDS_CLARIFICATION",
        "clarification_question": "Which databases should be compared?",
        "missing_fields": ["research_boundaries"],
    }
    fake = RecordingChatModel([invalid, invalid])

    with pytest.raises(ValueError, match="invalid structured output"):
        ScopeAgent(ModelRouter(fake, mode="fake")).assess("Compare databases")
