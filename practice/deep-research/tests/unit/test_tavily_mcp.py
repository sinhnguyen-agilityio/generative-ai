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
