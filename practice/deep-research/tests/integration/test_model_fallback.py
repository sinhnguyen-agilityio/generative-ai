from intelligence_collector.models.router import ModelRouter
from tests.fixtures.fake_models import FakeModel

def test_primary_failure_uses_fallback_and_records_event():
    primary = FakeModel(error_roles={"synthesis"})
    fallback = FakeModel(responses={"synthesis": "fallback draft"})
    router = ModelRouter(primary, fallback)
    assert router.invoke("synthesis", "draft") == "fallback draft"
    assert len(router.events) == 1 and fallback.calls
