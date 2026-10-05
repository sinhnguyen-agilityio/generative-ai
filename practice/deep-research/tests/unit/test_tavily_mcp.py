from types import SimpleNamespace

import pytest

from intelligence_collector.tools.mcp import TavilyMCP


def test_tavily_adapter_normalizes_results_without_exposing_key(monkeypatch):
    secret = "test-secret-value"
    adapter = TavilyMCP("https://mcp.tavily.com/mcp/", secret, max_calls=5)

    async def fake_search(query, max_results):
        return SimpleNamespace(
            structured_content={
                "results": [
                    {
                        "url": "https://docs.python.org/3/whatsnew/3.13.html",
                        "title": "What's New In Python 3.13",
                        "content": "Python 3.13 release information",
                        "score": 0.9,
                    }
                ]
            },
            content=[],
        )

    monkeypatch.setattr(adapter, "_search", fake_search)
    results = adapter.search("Python 3.13 release")

    assert results[0]["url"].startswith("https://docs.python.org/")
    assert "sufficient" not in results[0]  # sufficiency is the research agent's LLM decision
    assert adapter.requests == [
        {
            "query": "Python 3.13 release",
            "server": "tavily",
            "tool": "tavily_search",
        }
    ]
    assert secret not in repr(adapter)
    assert secret not in str(adapter.requests)


def test_tavily_adapter_requests_only_the_remaining_source_slots(monkeypatch):
    adapter = TavilyMCP("https://mcp.tavily.com/mcp/", "key", max_results=5)
    requested = []

    async def fake_search(query, max_results):
        requested.append(max_results)
        return SimpleNamespace(structured_content={"results": []}, content=[])

    monkeypatch.setattr(adapter, "_search", fake_search)
    adapter.search("q1")
    adapter.search("q2", max_results=2)
    adapter.search("q3", max_results=50)  # never above the adapter's own ceiling

    assert requested == [5, 2, 5]


def test_tavily_adapter_rejects_embedded_or_missing_credentials():
    with pytest.raises(ValueError, match="separately"):
        TavilyMCP(
            "https://mcp.tavily.com/mcp/?tavilyApiKey=embedded",
            "separate-key",
        )
    with pytest.raises(ValueError, match="non-empty"):
        TavilyMCP("https://mcp.tavily.com/mcp/", "")


def test_tavily_adapter_keeps_the_snippet_and_the_raw_page_apart(monkeypatch):
    adapter = TavilyMCP("https://mcp.tavily.com/mcp/", "key")

    async def fake_search(query, max_results):
        return SimpleNamespace(
            structured_content={
                "results": [
                    {
                        "url": "https://docs.test/page",
                        "title": "Page",
                        "content": "Short snippet.",
                        "raw_content": "# Page\n\nThe whole page text.",
                        "score": 0.8,
                    },
                    {"url": "https://docs.test/no-raw", "title": "No raw", "content": "Only."},
                ]
            },
            content=[],
        )

    monkeypatch.setattr(adapter, "_search", fake_search)
    page, no_raw = adapter.search("q")

    assert page["content"] == "Short snippet."  # the snippet is never replaced by the page
    assert page["raw_content"] == "# Page\n\nThe whole page text."
    assert no_raw["raw_content"] == ""


def test_tavily_search_asks_for_raw_content(monkeypatch):
    import contextlib

    from intelligence_collector.tools import mcp as mcp_module

    calls = []

    @contextlib.asynccontextmanager
    async def fake_client(url):
        yield ("read", "write")

    class FakeSession:
        def __init__(self, read, write):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            return None

        async def call_tool(self, name, arguments, read_timeout_seconds=None):
            calls.append((name, arguments))
            return SimpleNamespace(structured_content={"results": []}, content=[])

    monkeypatch.setattr(mcp_module, "streamable_http_client", fake_client)
    monkeypatch.setattr(mcp_module, "ClientSession", FakeSession)

    TavilyMCP("https://mcp.tavily.com/mcp/", "key").search("q", max_results=2)
    TavilyMCP("https://mcp.tavily.com/mcp/", "key", include_raw_content=False).search("q")

    assert calls == [
        ("tavily_search", {"query": "q", "max_results": 2, "include_raw_content": True}),
        ("tavily_search", {"query": "q", "max_results": 5, "include_raw_content": False}),
    ]
