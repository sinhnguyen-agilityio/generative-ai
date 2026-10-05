"""Embedding reuse between passage selection and the vector index.

Passage selection embeds every chunk of a fetched page to rank it against the research question
(``agents/passage_selector.py``); a moment later the worker indexes the same chunks in Chroma.
``CachingEmbeddings`` keeps the vectors selection computed, so indexing hands them to Chroma
instead of embedding the same texts again.
"""

from __future__ import annotations

import hashlib
import threading
from array import array
from collections import OrderedDict

# Vectors are held only until the worker indexes them, normally seconds later; the bound protects
# memory if a page is selected but never indexed (a failed task).
DEFAULT_MAX_ENTRIES = 5_000


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachingEmbeddings:
    """Wraps LangChain embeddings. ``remember`` stores document vectors; ``embed_documents``
    returns a remembered vector once (and forgets it) and embeds only the texts it does not hold.
    Thread-safe: parallel research sub-agents share one instance."""

    def __init__(self, embeddings, *, max_entries: int = DEFAULT_MAX_ENTRIES):
        self.inner = embeddings
        self.max_entries = max_entries
        self._vectors: OrderedDict[str, array] = OrderedDict()
        self._lock = threading.Lock()

    def remember(self, texts: list[str], vectors: list[list[float]]) -> None:
        with self._lock:
            for text, vector in zip(texts, vectors):
                self._vectors[_key(text)] = array("f", vector)
            while len(self._vectors) > self.max_entries:
                self._vectors.popitem(last=False)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with self._lock:
            held = {i: self._vectors.pop(_key(text), None) for i, text in enumerate(texts)}
        missing = [i for i, vector in held.items() if vector is None]
        computed = iter(self.inner.embed_documents([texts[i] for i in missing]) if missing else [])
        return [
            next(computed) if held[i] is None else held[i].tolist() for i in range(len(texts))
        ]

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)
