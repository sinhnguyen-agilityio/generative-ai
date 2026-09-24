from intelligence_collector.observability.tracing import Tracer
def test_tracer_correlates_spans_without_external_client():
    tracer=Tracer()
    with tracer.span("node", "run", "thread", stage="Planning") as span:
        assert span.run_id == "run"
    assert tracer.spans[0].thread_id == "thread"
