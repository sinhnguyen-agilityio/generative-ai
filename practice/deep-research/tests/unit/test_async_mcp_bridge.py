import asyncio
from contextlib import contextmanager
from contextvars import ContextVar

from intelligence_collector.tools.async_bridge import run_async


def test_sync_bridge_inside_event_loop_and_worker_preserves_context():
    marker = ContextVar("marker", default="missing")

    async def response():
        await asyncio.sleep(0)
        return marker.get()

    async def scenario():
        marker.set("research-trace")
        assert run_async(response) == "research-trace"
        assert await asyncio.to_thread(run_async, response) == "research-trace"

    asyncio.run(scenario())


def test_fetch_operation_trace_contains_redacted_url(stores):
    from intelligence_collector.services.operation_context import operation_tracer
    from intelligence_collector.services.workflow import DurableOperations
    spans = []

    class Tracer:
        @contextmanager
        def span(self, *args, **kwargs):
            spans.append((args, kwargs))
            yield

    token = operation_tracer.set(Tracer())
    try:
        DurableOperations(stores[0]).call("run", "tavily_fetch",
            {"url": "https://source.test/page?tavilyApiKey=private-key"}, lambda: {"content": "original"})
    finally:
        operation_tracer.reset(token)
    assert spans[0][0][0] == "operation.tavily_fetch"
    assert "https://source.test/page" in spans[0][1]["input"]["arguments"]["url"]
    assert "private-key" not in str(spans)
