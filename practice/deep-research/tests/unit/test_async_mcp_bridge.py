import asyncio
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
    from intelligence_collector.services.workflow import DurableOperations
    from tests.fixtures.langfuse_capture import LangfuseCapture

    capture = LangfuseCapture()
    with capture.tracer.execution(run_id="run", thread_id="thread", kind="start"):
        DurableOperations(stores[0]).call(
            "run",
            "tavily_fetch",
            {"url": "https://source.test/page?tavilyApiKey=private-key"},
            lambda: {"content": "original"},
        )

    span = capture.named("operation.tavily_fetch")[0]
    assert span.input["arguments"]["url"] == "https://source.test/page?tavilyApiKey=[REDACTED]"
    assert "private-key" not in repr(capture.exporter.get_finished_spans()[0].attributes)
