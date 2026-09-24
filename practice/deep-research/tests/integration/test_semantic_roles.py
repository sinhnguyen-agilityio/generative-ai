from intelligence_collector.agents.llm_roles import SemanticRoles
from intelligence_collector.models.router import ModelRouter
from tests.fixtures.fake_models import FakeModel

def test_all_semantic_roles_use_structured_router():
    responses = {
        "planner": {"todos": []}, "supervisor": {"coverage_complete": True, "research_complete": True},
        "worker": {"queries": ["broad"], "stop_condition": "enough evidence"},
        "synthesizer": {"report_markdown": "draft"},
        "verifier": {"passed": True}, "fixer": {"action": "revise", "reason": "draft defect"},
    }
    fake = FakeModel(responses=responses); roles = SemanticRoles(ModelRouter(fake, fake, mode="fake"))
    assert roles.planner("brief") and roles.supervisor("context") and roles.worker("question")
    assert roles.synthesizer("evidence") and roles.verifier("draft") and roles.fixer("feedback")
    assert {call[0] for call in fake.calls} == {"planner", "supervisor", "worker", "synthesizer", "verifier", "fixer"}
