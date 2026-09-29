from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from intelligence_collector.models.router import ModelRouter
from tests.fixtures.fake_models import FakeModel


class _Draft(BaseModel):
    text: str


def test_primary_failure_uses_fallback():
    primary = FakeModel(error_roles={"synthesis"})
    fallback = FakeModel(responses={"synthesis": {"text": "fallback draft"}})
    router = ModelRouter(primary, fallback)
    result = router.invoke_messages("synthesis", [HumanMessage(content="draft")], _Draft)
    assert result == _Draft(text="fallback draft")
    assert primary.calls and fallback.calls
