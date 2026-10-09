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


def normalize_url(url: str) -> str:
    """Normalize page identity by removing fragments, tracking parameters, and trailing slashes.

    Args:
        url: Page URL, possibly with whitespace or tracking parameters.

    Returns:
        URL with lowercase scheme/host and non-utm query parameters preserved.
    """
    parts = urlsplit(url.strip())
    params = [(k, v) for k, v in parse_qsl(
        parts.query, keep_blank_values=True) if not k.lower().startswith("utm_")]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), urlencode(params), ""))


def clean(text: str) -> str:
    """Remove repeated lines and common navigation, link, and banner noise.

    Args:
        text: Raw page text to clean.

    Returns:
        Remaining content joined with newlines in its original order.
    """
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
    """Prepare unique usable pages, falling back to snippets when cleaned text is too short.

    Args:
        results: Normalized search results with raw_content, content, URL, title, and score.

    Returns:
        Pages with url, title, text, snippet, full_text, and score; short results are dropped.
    """
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
    """Group passages by URL and label each page with its source number.

    Separate passages of one page with ellipses and pages with a Markdown rule.
    The full-text label describes the source metadata, not the length of the displayed excerpt.

    Args:
        passages: URL/text passages in the order they should be displayed.
        sources: Registry keyed by URL with id, title, and full_text metadata for each passage.

    Returns:
        Citation-numbered evidence text, or "No results." when there are no passages.
    """
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
    """Initialize the embedding client for the configured provider.

    Args:
        model: Embedding model identifier.
        provider: openai selects OpenAI; other values select Google GenAI.

    Returns:
        LangChain embedding client.
    """
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=model)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=model)


class PageStore:
    """Result pages in a local Chroma collection, chunked and embedded, searchable per run."""

    def __init__(self, path: str, model: str, provider: str):
        """Open a persistent Chroma collection and configure embeddings and page chunking.

        Args:
            path: Directory for the local Chroma database.
            model: Embedding model determining the collection and vector dimensions.
            provider: Embedding provider used to create the client and collection name.
        """
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
        """Chunk and embed pages, then upsert them into the current run.

        Chunk IDs depend on run, URL, and chunk index, so repeated pages replace existing
        chunks with matching IDs even when another topic discovers them.

        Args:
            pages: Prepared pages including text and source metadata.
            run_id: Research thread used to isolate indexed evidence.
            topic: Delegated topic recorded in chunk metadata.
        """
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
        """Retrieve similar passages from the requested pages of one run.

        Args:
            text: Query embedded for similarity search.
            run_id: Thread identifier restricting the collection search.
            urls: Page URLs eligible for retrieval.
            k: Maximum returned passages, also subject to the per-URL chunk cap.

        Returns:
            Up to k URL/text passages in retrieval order.
        """
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
