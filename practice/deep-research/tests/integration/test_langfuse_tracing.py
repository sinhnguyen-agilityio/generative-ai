from intelligence_collector.observability.tracing import Tracer
def test_trace_events_are_correlated_without_external_service():
    tracer = Tracer()
    with tracer.span("mcp.search", "run", "thread", task_id="task"):
        pass
    assert tracer.spans[0].run_id == "run"
    assert tracer.spans[0].thread_id == "thread"
