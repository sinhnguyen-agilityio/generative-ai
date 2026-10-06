"""Passage selection: the parts of each fetched page that bear on the research question.

Tavily returns every page's full text (``raw_content``). Putting whole pages into the researcher's
context would overflow it, and summarizing each page with an LLM costs one large model call per
page. Instead, as GPT Researcher's context compressor does, each page is cleaned and split into
the chunks the vector index stores (``storage/evidence_index.chunk_texts``), the chunks are
ranked against the research question and the search query, and the best few are kept VERBATIM,
in page order. They become the result's ``content``: what the researcher, the compressor, the
report writer and the verifiers read, so every gate checks real page text. The cleaned page stays
in ``raw_content`` for the VFS and the vector index only.

Ranking uses the injected embeddings (cosine similarity). Without embeddings, or when embedding
fails, it falls back to word overlap: an embedding problem never stops research. Vectors computed
here are remembered (``CachingEmbeddings``), so indexing the same chunks does not embed them again.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from intelligence_collector.models.redaction import redact_error
from intelligence_collector.observability.tracing import traced
from intelligence_collector.storage.evidence_index import chunk_texts
from intelligence_collector.tools.page_content import clean_page, usable_page

logger = logging.getLogger(__name__)

# Passages kept per page (each at most ``CHUNK_CHARS``): with three pages per task, the writer
# reads about 9,000 characters of page text per research question.
MAX_PASSAGES_PER_PAGE = 3
PASSAGE_SEPARATOR = "\n\n[…]\n\n"

_WORD = re.compile(r"[a-z0-9]+")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _overlap(query: str, texts: list[str]) -> list[float]:
    """Share of the query's words found in each text (the offline fallback ranking)."""
    terms = {word for word in _WORD.findall(query.casefold()) if len(word) > 2}
    if not terms:
        return [0.0] * len(texts)
    return [len(terms & set(_WORD.findall(text.casefold()))) / len(terms) for text in texts]


def _passages(chunks: list[str], header: str, scores: list[float], keep: int) -> str:
    """The `keep` best chunks, in page order, without the title header used for ranking."""
    best = sorted(sorted(range(len(chunks)), key=lambda i: -scores[i])[:keep])
    return PASSAGE_SEPARATOR.join(chunks[i].removeprefix(header) for i in best)


class PassageSelector:
    """Keeps the most relevant verbatim passages of each fetched page."""

    def __init__(self, embeddings=None, *, max_passages: int = MAX_PASSAGES_PER_PAGE):
        self.embeddings = embeddings
        self.max_passages = max(1, max_passages)

    def _scores(self, query: str, texts: list[str]) -> list[float]:
        """Relevance of each chunk text to `query`: cosine similarity of embeddings, or word
        overlap without embeddings or when embedding fails."""
        if self.embeddings is None or not texts:
            return _overlap(query, texts)
        try:
            with traced("passage.select", as_type="embedding", metadata={"chunks": len(texts)}):
                vectors = self.embeddings.embed_documents(texts)
                target = self.embeddings.embed_query(query)
        except Exception as exc:
            # No traceback: embedding client errors can echo the request URL and its API key.
            logger.warning("Passage ranking fell back to word overlap: %s", redact_error(exc))
            return _overlap(query, texts)
        remember = getattr(self.embeddings, "remember", None)
        if remember is not None:
            remember(texts, vectors)
        return [_cosine(vector, target) for vector in vectors]

    def read_pages(self, query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Clean each row's raw page and replace its snippet ``content`` with the page's most
        relevant passages for `query` (the research question and the search query).

        Rows keep their order. ``raw_content`` becomes the cleaned page, or "" when the page is
        unusable (the snippet then stays the evidence); ``snippet`` keeps Tavily's excerpt
        whenever passages replaced it. All chunks of one search are ranked in one embedding call.
        """
        read: list[dict[str, Any]] = []
        pending: list[tuple[int, list[str], str]] = []  # (row position, chunk texts, header)
        for row in rows:
            page, truncated = clean_page(str(row.get("raw_content") or ""))
            if not usable_page(page):
                read.append({**row, "raw_content": ""})
                continue
            title = str(row.get("title") or "")
            read.append(
                {
                    **row,
                    "raw_content": page,
                    "raw_truncated": truncated,
                    "snippet": str(row.get("content", "")),
                }
            )
            header = f"{title}\n\n" if title else ""
            pending.append((len(read) - 1, chunk_texts(title, page), header))
        texts = [
            text
            for _, chunks, _ in pending
            if len(chunks) > self.max_passages
            for text in chunks
        ]
        scores = iter(self._scores(query, texts))
        for position, chunks, header in pending:
            ranked = (
                [next(scores) for _ in chunks]
                if len(chunks) > self.max_passages
                else [0.0] * len(chunks)  # short page: every chunk is kept
            )
            read[position]["content"] = _passages(chunks, header, ranked, self.max_passages)
        return read
