from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.parse import urlencode, urlsplit, urlunsplit

import anyio
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class MCPResearchPort(Protocol):
    """Approved boundary through which research workers access external sources."""

    max_calls: int

    def search(
        self,
        query: str,
        *,
        server: str = "controlled",
        tool: str = "search",
    ) -> list[dict[str, Any]]: ...


class ControlledMCP:
    """Deterministic MCP adapter for offline execution and contract tests."""

    def __init__(self, sources, max_calls: int = 5):
        is_sequence = bool(sources) and isinstance(sources[0], list)
        self._responses = list(sources) if is_sequence else [list(sources)]
        self.calls = 0
        self.max_calls = min(max(1, max_calls), 5)
        self.server_name = "controlled"
        self.tool_name = "search"
        self.requests: list[dict[str, str]] = []

    def search(self, query, *, server="controlled", tool="search"):
        if self.calls >= self.max_calls:
            raise RuntimeError("search budget exhausted")
        response_index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        self.requests.append({"query": query, "server": server, "tool": tool})
        return list(self._responses[response_index])


class TavilyMCP:
    """Secret-safe synchronous adapter for Tavily's streamable HTTP MCP server."""

    server_name = "tavily"
    tool_name = "tavily_search"

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        *,
        max_calls: int = 5,
        max_results: int = 5,
        timeout_seconds: float = 30.0,
    ):
        if not api_key.strip():
            raise ValueError("Tavily MCP requires a non-empty API key")
        parsed = urlsplit(endpoint.strip())
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Tavily MCP endpoint must be an HTTPS URL")
        if "tavilyApiKey" in parsed.query:
            raise ValueError("configure the Tavily API key separately from the endpoint")
        query = urlencode({"tavilyApiKey": api_key.strip()})
        self._url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))
        self.calls = 0
        self.max_calls = min(max(1, max_calls), 5)
        self.max_results = min(max(1, max_results), 10)
        self.timeout_seconds = timeout_seconds
        self.requests: list[dict[str, str]] = []

    def __repr__(self) -> str:
        return (
            f"TavilyMCP(server_name={self.server_name!r}, "
            f"max_calls={self.max_calls!r})"
        )

    async def _search(self, query: str):
        async with streamable_http_client(self._url) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                return await session.call_tool(
                    self.tool_name,
                    {"query": query, "max_results": self.max_results},
                    read_timeout_seconds=self.timeout_seconds,
                )

    @staticmethod
    def _result_payload(result) -> list[dict[str, Any]]:
        payload = getattr(result, "structured_content", None)
        if not payload:
            text = "\n".join(
                block.text for block in getattr(result, "content", [])
                if getattr(block, "type", "") == "text"
            )
            try:
                payload = json.loads(text)
            except (TypeError, json.JSONDecodeError):
                payload = {"results": []}
        rows = payload.get("results", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            return []
        normalized = []
        for row in rows:
            if not isinstance(row, dict) or not str(row.get("url", "")).strip():
                continue
            content = row.get("content") or row.get("raw_content") or row.get("title") or ""
            normalized.append({
                "url": str(row["url"]),
                "content": str(content),
                "title": str(row.get("title", "")),
                "score": row.get("score"),
                "sufficient": True,
            })
        return normalized

    def search(self, query, *, server="tavily", tool="tavily_search"):
        if self.calls >= self.max_calls:
            raise RuntimeError("search budget exhausted")
        if not str(query).strip():
            raise ValueError("Tavily MCP search query must not be blank")
        self.calls += 1
        self.requests.append({"query": str(query), "server": server, "tool": tool})
        return self._result_payload(anyio.run(self._search, str(query)))
