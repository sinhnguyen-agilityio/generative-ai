from intelligence_collector.models.router import ModelRouter
def test_model_router_falls_back_once():
    events=[]
    def primary(role, prompt): raise RuntimeError("timeout")
    def fallback(role, prompt): return "fallback"
    router=ModelRouter(primary, fallback, on_fallback=events.append)
    assert router.invoke("scope", "hello") == "fallback"
    assert len(router.events) == len(events) == 1
