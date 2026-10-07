"""Web search through Tavily's MCP server, and a Chroma store for the scraped pages."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from urllib.parse import urlencode

import anyio
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)

# Characters of each page shown to the researcher; the full page goes to Chroma.
EXCERPT_CHARS = 1500


class TavilySearch:
    """Calls the ``tavily_search`` tool of Tavily's streamable-HTTP MCP server."""

    def __init__(self, url: str, api_key: str, max_results: int = 5, timeout: float = 30.0):
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required")
        self._url = f"{url.rstrip('/')}/?{urlencode({'tavilyApiKey': api_key})}"
        self.max_results = max_results
        self.timeout = timeout

    async def _call(self, query: str):
        async with streamable_http_client(self._url) as (read, write, *_):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool(
                    "tavily_search",
                    {"query": query, "max_results": self.max_results, "include_raw_content": True},
                    read_timeout_seconds=self.timeout,
                )

    def search(self, query: str) -> list[dict]:
        """Results as ``{url, title, content, raw_content}`` dicts."""
        try:
            result = anyio.run(self._call, query)
        except Exception as exc:
            # The URL carries the API key and transport errors quote it: raise a clean error,
            # without the original as context, so it never reaches the model, logs or traces.
            raise RuntimeError(f"Tavily search failed ({type(exc).__name__})") from None
        payload = getattr(result, "structured_content", None)
        if not payload:
            text = "".join(b.text for b in result.content if getattr(b, "type", "") == "text")
            payload = json.loads(text or "{}")
        rows = payload.get("results", []) if isinstance(payload, dict) else payload
        return [
            {
                "url": row["url"],
                "title": row.get("title") or row["url"],
                "content": row.get("content") or "",
                "raw_content": row.get("raw_content") or "",
            }
            for row in rows
            if isinstance(row, dict) and row.get("url")
        ]


def format_results(results: list[dict]) -> str:
    if not results:
        return "No results."
    blocks = []
    for row in results:
        excerpt = (row["raw_content"] or row["content"])[:EXCERPT_CHARS]
        blocks.append(f"Title: {row['title']}\nURL: {row['url']}\nSummary: {row['content']}\nExcerpt: {excerpt}")
    return "\n\n---\n\n".join(blocks)


def _embeddings(model: str, provider: str):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=model)


class PageStore:
    """Persists scraped pages in a local Chroma collection, chunked and embedded."""

    def __init__(self, path: str, model: str, provider: str):
        import chromadb

        Path(path).mkdir(parents=True, exist_ok=True)
        self._embeddings = _embeddings(model, provider)
        # One collection per embedding model: vectors of different sizes cannot share one.
        name = f"pages_{provider}_{model}".replace("/", "-")[:60]
        self._collection = chromadb.PersistentClient(path=path).get_or_create_collection(name)
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)

    def add(self, results: list[dict], *, run_id: str, topic: str) -> None:
        ids, texts, metadatas = [], [], []
        for row in results:
            for i, chunk in enumerate(self._splitter.split_text(row["raw_content"] or row["content"])):
                ids.append(hashlib.sha1(f"{run_id}|{row['url']}|{i}".encode()).hexdigest())
                texts.append(chunk)
                metadatas.append({"run_id": run_id, "topic": topic, "url": row["url"], "title": row["title"]})
        if not ids:
            return
        try:
            vectors = self._embeddings.embed_documents(texts)
            self._collection.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        except Exception:  # storage is an archive; research continues without it
            logger.exception("Could not store %d chunks in Chroma", len(ids))
