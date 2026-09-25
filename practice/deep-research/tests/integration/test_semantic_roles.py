from intelligence_collector.agents.llm_roles import SemanticRoles
from intelligence_collector.agents.prompts import ROLE_TEMPLATES
from intelligence_collector.models.router import ModelRouter
from tests.fixtures.fake_models import FakeModel


def _responses():
    return {
        "planner": {"todos": [{"research_question": "Find primary evidence"}]},
        "supervisor": {
            "coverage_complete": True,
            "research_complete": True,
        },
        "worker": {
            "queries": ["broad"],
            "stop_condition": "enough evidence",
        },
        "synthesizer": {
            "report_markdown": "draft",
            "claim_evidence_ids": {"claim": ["evidence-1"]},
        },
        "verifier": {"passed": True},
        "fixer": {"action": "revise", "reason": "draft defect"},
    }

def test_all_semantic_roles_use_structured_router():
    fake = FakeModel(responses=_responses()); roles = SemanticRoles(ModelRouter(fake, fake, mode="fake"))
    assert roles.planner("brief") and roles.supervisor("context") and roles.worker("question")
    assert roles.synthesizer("evidence") and roles.verifier("draft") and roles.fixer("feedback")
    assert {call[0] for call in fake.calls} == {"planner", "supervisor", "worker", "synthesizer", "verifier", "fixer"}


def test_every_semantic_role_prompt_is_versioned_and_instructive():
    expected = {
        "planner": "planner.v1",
        "supervisor": "supervisor.v1",
        "worker": "worker.v1",
        "synthesizer": "synthesis.v1",
        "verifier": "verifier.v1",
        "fixer": "fixer.v1",
    }
    variables = {
        "planner": {"brief": "brief"},
        "supervisor": {"context": "context"},
        "worker": {"question": "question"},
        "synthesizer": {"evidence": "evidence"},
        "verifier": {"draft": "draft"},
        "fixer": {"feedback": "feedback"},
    }

    for role, version in expected.items():
        rendered = "\n".join(
            str(message.content)
            for message in ROLE_TEMPLATES[role].format_messages(**variables[role])
        )
        assert version in rendered
        assert "Return only" in rendered
