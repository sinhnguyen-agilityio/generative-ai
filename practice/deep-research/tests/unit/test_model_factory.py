from intelligence_collector.config import RoleModelSettings
from intelligence_collector.models.factory import ModelFactory
from intelligence_collector.models.router import MAX_MODEL_ATTEMPTS


def test_reasoning_router_gets_every_retry_attempt_the_router_allows(monkeypatch):
    """Groq's strict structured-output decoding for gpt-oss is intermittently flaky server-side
    (json_validate_failed on a fraction of otherwise-identical requests) -- retrying is the only
    lever available for that, so the reasoning tier (which carries the report-judge "verify"
    call) should get the router's maximum retry budget, not the default."""
    monkeypatch.setattr(
        "intelligence_collector.models.factory.init_chat_model", lambda model, **kwargs: object()
    )
    settings = RoleModelSettings(reasoning_model="r-model", standard_model="s-model")

    routers = ModelFactory().create_routers(settings)

    assert routers.reasoning.max_attempts == MAX_MODEL_ATTEMPTS
    assert routers.standard.max_attempts == 2
