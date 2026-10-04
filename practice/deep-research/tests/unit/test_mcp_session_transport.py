from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("operation", ["search", "fetch", "storage"])
def test_transport_callback_is_not_passed_as_session_timeout(monkeypatch, operation):
    from intelligence_collector.tools import mcp as research
    from intelligence_collector.storage import mcp_vector_index as storage

    module = storage if operation == "storage" else research
    reader, writer = object(), object()
    calls = []

    @asynccontextmanager
    async def transport(url):
        yield reader, writer, lambda: "session-id"

    class Session:
        def __init__(self, read, write, *, read_timeout_seconds):
            assert read is reader and write is writer
            assert isinstance(read_timeout_seconds, timedelta)
            assert read_timeout_seconds.total_seconds() == 30

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, tool, arguments, *, read_timeout_seconds):
            assert read_timeout_seconds == timedelta(seconds=30)
            if operation == "search":
                assert arguments == {
                    "query": "question",
                    "max_results": 3,
                    "include_raw_content": True,
                }
            calls.append(tool)
            return SimpleNamespace(
                isError=False,
                structured_content={
                    "results": [{"url": "https://source.test/", "raw_content": "original"}]
                },
                content=[],
            )

    monkeypatch.setattr(module, "streamable_http_client", transport)
    monkeypatch.setattr(module, "ClientSession", Session)
    if operation == "storage":
        storage.MCPStorageTransport("http://localhost:8000/mcp", timeout_seconds=30).call(
            "embed", {}
        )
    else:
        adapter = research.TavilyMCP("https://mcp.tavily.com/mcp/", "fake-key", timeout_seconds=30)
        if operation == "search":
            adapter.search("question")
        else:
            adapter.fetch("https://source.test/")
    assert len(calls) == 1
