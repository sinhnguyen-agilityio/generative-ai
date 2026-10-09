"""Web search through Tavily's MCP server, and a Chroma store that indexes the result pages and
returns the passages most relevant to a query."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

# cleaned page text shorter than this is not a usable page; use the snippet
MIN_FULL_TEXT = 500
MIN_TEXT = 200  # results with less text than this are dropped
MAX_CHUNKS_PER_URL = 2  # so one page cannot fill all retrieved passages

_LINK_ONLY = re.compile(r"^(?:[-*|]\s*)?!?\[[^\]]*\]\([^)]*\)\s*\|?$")
_NOISE = re.compile(
    r"cookie|subscribe|sign (?:in|up)|log in|all rights reserved|privacy policy|terms of (?:use|service)", re.I
)


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
                    {"query": query, "max_results": self.max_results,
                        "include_raw_content": True},
                    read_timeout_seconds=self.timeout,
                )

    async def search(self, query: str) -> list[dict]:
        """Results as ``{url, title, content, raw_content, score}`` dicts."""
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


def normalize_url(url: str) -> str:
    """One key per page: no fragment, no utm_* parameters, no trailing slash, lower-case host."""
    parts = urlsplit(url.strip())
    params = [(k, v) for k, v in parse_qsl(
        parts.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(params), ""))


def clean(text: str) -> str:
    """Drop page noise: blank and repeated lines, link-only lines, short menu items, cookie banners."""
    seen, kept = set(), []
    for line in text.splitlines():
        s = line.strip()
        if not s or s in seen or _LINK_ONLY.match(s):
            continue
        if len(s) < 25 and not s.startswith("#") and not any(c.isdigit() for c in s):
            continue
        if len(s) < 120 and _NOISE.search(s):
            continue
        seen.add(s)
        kept.append(s)
    return "\n".join(kept)


def prepare_pages(results: list[dict]) -> list[dict]:
    """Search results as pages ``{url, title, text, snippet, full_text, score}``. A page whose
    cleaned text is too short falls back to Tavily's snippet; a result with too little text is dropped,
    and so is a result whose normalized URL an earlier result already has."""
    pages = {}
    for result in results:
        url = normalize_url(result["url"])
        text, snippet = clean(result["raw_content"]), result["content"].strip()
        full_text = len(text) >= MIN_FULL_TEXT
        if not full_text:
            text = snippet
        if url not in pages and len(text) >= MIN_TEXT:
            pages[url] = {
                # One line: a title is rendered into source lines, so it must not break them.
                "url": url, "title": " ".join(result["title"].split()) or url, "text": text,
                "snippet": snippet, "full_text": full_text, "score": result["score"],
            }
    return list(pages.values())


def format_passages(passages: list[dict], sources: dict[str, dict]) -> str:
    """Passages grouped by page, each page headed by its source number."""
    grouped: dict[str, list[str]] = {}
    for passage in passages:
        grouped.setdefault(passage["url"], []).append(passage["text"])
    blocks = []
    for url, texts in grouped.items():
        source = sources[url]
        kind = "full text" if source["full_text"] else "snippet only"
        blocks.append(
            f"[{source['id']}] {source['title']} — {url} ({kind})\n" + "\n...\n".join(texts))
    return "\n\n---\n\n".join(blocks) or "No results."


def _embeddings(model: str, provider: str):
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=model)


class PageStore:
    """Result pages in a local Chroma collection, chunked and embedded, searchable per run."""

    def __init__(self, path: str, model: str, provider: str):
        import chromadb

        Path(path).mkdir(parents=True, exist_ok=True)
        self._embeddings = _embeddings(model, provider)
        # One collection per embedding model: vectors of different sizes cannot share one.
        name = f"pages_cos_{provider}_{model}".replace("/", "-")[:60]
        self._collection = chromadb.PersistentClient(path=path).get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)

    def add(self, pages: list[dict], *, run_id: str, topic: str) -> None:
        ids, texts, metadatas = [], [], []
        for page in pages:
            for i, chunk in enumerate(self._splitter.split_text(page["text"])):
                # No topic in the id: a page has one set of chunks per run, whichever topic found it.
                ids.append(hashlib.sha1(
                    f"{run_id}|{page['url']}|{i}".encode()).hexdigest())
                texts.append(chunk)
                metadatas.append({
                    "run_id": run_id, "topic": topic, "url": page["url"], "title": page["title"],
                    "full_text": page["full_text"], "score": page["score"],
                })
        if ids:
            vectors = self._embeddings.embed_documents(texts)
            self._collection.upsert(
                ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    def query(self, text: str, *, run_id: str, urls: list[str], k: int) -> list[dict]:
        """The ``k`` chunks of these pages most similar to ``text``, as ``{url, text}``."""
        result = self._collection.query(
            query_embeddings=[self._embeddings.embed_query(text)],
            n_results=k * MAX_CHUNKS_PER_URL,
            where={"$and": [{"run_id": run_id}, {"url": {"$in": urls}}]},
        )
        passages, per_url = [], {}
        for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
            if per_url.get(meta["url"], 0) < MAX_CHUNKS_PER_URL:
                per_url[meta["url"]] = per_url.get(meta["url"], 0) + 1
                passages.append({"url": meta["url"], "text": doc})
        return passages[:k]
