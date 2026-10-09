"""Tavily search definitions."""

from __future__ import annotations

import json
from urllib.parse import urlencode

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


class TavilySearch:
    """Calls the ``tavily_search`` tool of Tavily's streamable-HTTP MCP server."""

    def __init__(self, url: str, api_key: str, max_results: int = 5, timeout: float = 30.0):
        """Configure the Tavily MCP endpoint and search request limits.

        Args:
            url: Base streamable-HTTP MCP URL.
            api_key: Tavily credential added to the endpoint URL.
            max_results: Maximum requested results per query.
            timeout: Tool read timeout in seconds.

        Raises:
            ValueError: The API key is empty.
        """
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required")
        self._url = f"{url.rstrip('/')}/?{urlencode({'tavilyApiKey': api_key})}"
        self.max_results = max_results
        self.timeout = timeout

    async def _call(self, query: str):
        """Open an MCP session and execute one Tavily search request.

        Args:
            query: Web search query.

        Returns:
            MCP tool result containing structured data or text content.
        """
        async with streamable_http_client(self._url) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(
                    "tavily_search",
                    {"query": query, "max_results": self.max_results,
                        "include_raw_content": True},
                    read_timeout_seconds=self.timeout,
                )

    async def search(self, query: str) -> list[dict]:
        """Search Tavily and normalize usable result records.

        Transport errors are replaced with a clean message so the credential-bearing
        endpoint URL is not exposed through the original exception.

        Args:
            query: Web search query.

        Returns:
            Result dictionaries with url, title, content, raw_content, and score.

        Raises:
            RuntimeError: The MCP request fails.
            json.JSONDecodeError: Text-only tool content is not valid JSON.
        """
        try:
            result = await self._call(query)
        except Exception as exc:
            # The URL carries the API key and transport errors quote it: raise a clean error,
            # without the original as context, so it never reaches the model, logs or traces.
            raise RuntimeError(
                f"Tavily search failed ({type(exc).__name__})") from None
        payload = getattr(result, "structured_content", None)
        if not payload:
            text = "".join(b.text for b in result.content if getattr(
                b, "type", "") == "text")
            payload = json.loads(text or "{}")
        rows = payload.get("results", []) if isinstance(
            payload, dict) else payload
        return [
            {
                "url": row["url"],
                "title": row.get("title") or row["url"],
                "content": row.get("content") or "",
                "raw_content": row.get("raw_content") or "",
                "score": float(row.get("score") or 0.0),
            }
            for row in rows
            if isinstance(row, dict) and row.get("url")
        ]
