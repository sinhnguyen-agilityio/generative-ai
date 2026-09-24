from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.contracts import ScopeDecision
from intelligence_collector.models.router import ModelRouter
from tests.fixtures.fake_models import FakeModel

def test_scope_llm_asks_one_focused_question_for_ambiguous_request():
    fake = FakeModel(responses={"scope": {"decision": "NEEDS_CLARIFICATION", "clarification_question": "Which databases and workload?", "missing_fields": ["research_boundaries"], "reason": "too broad"}})
    result = ScopeAgent(ModelRouter(fake, fake, mode="fake")).assess("Compare databases")
    assert result.decision == ScopeDecision.NEEDS_CLARIFICATION
    assert result.clarification_question

def test_scope_llm_returns_clear_brief():
    payload = {"decision": "SCOPE_CLEAR", "objective": "Compare PostgreSQL and SQLite", "research_boundaries": "local Python apps", "constraints": "technical sources", "success_criteria": "claims have evidence", "deliverable_format": "technical report"}
    fake = FakeModel(responses={"scope": payload})
    result = ScopeAgent(ModelRouter(fake, fake, mode="fake")).assess("Compare PostgreSQL and SQLite")
    assert result.decision == ScopeDecision.SCOPE_CLEAR and result.success_criteria
