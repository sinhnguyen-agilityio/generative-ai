"""Hybrid passage lookup over a run's stored pages (spec 002, FR-013d–f).

A lookup answers one evidence point from passages already captured in the run, before anyone
searches the web again. Every query (the evidence point with its task question, plus a few
alternative phrasings) is ranked twice: by meaning (the vector index) and by exact keywords (an
in-house Okapi BM25 over the run's chunks, so numbers, names and version strings are found). The
rankings are fused with Reciprocal Rank Fusion and the top passages are returned; there is no
re-ranking model and no relevance threshold. A lookup only reads the index.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from intelligence_collector.contracts import IndexedChunk
from intelligence_collector.models.redaction import redact_error
from intelligence_collector.observability.tracing import traced
from intelligence_collector.ports import VectorIndex

DEFAULT_TOP_K = 5
DEFAULT_CANDIDATES = 10
RRF_K = 60
BM25_K1 = 1.5
BM25_B = 0.75
_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    """Okapi BM25 over a fixed list of chunk texts."""

    def __init__(self, texts: list[str], *, k1: float = BM25_K1, b: float = BM25_B):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(text)) for text in texts]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.docs else 0.0
        frequency: Counter[str] = Counter()
        for doc in self.docs:
            frequency.update(doc.keys())
        total = len(self.docs)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        result = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            result.append(score)
        return result

    def top(self, query: str, limit: int) -> list[int]:
        """Indexes of the best-scoring documents with a positive score, best first."""
        scored = [(score, index) for index, score in enumerate(self.scores(query)) if score > 0]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [index for _, index in scored[:limit]]


@dataclass
class LookupResult:
    """The passages a lookup returned, best first, with how each ranking placed them."""

    chunks: list[IndexedChunk] = field(default_factory=list)
    ranks: list[dict] = field(default_factory=list)
    fallback: str = "none"
    error: str = ""


class PassageLookup:
    """Multi-query hybrid lookup over one run's indexed passages (contracts/passage-lookup.md)."""

    def __init__(
        self,
        index: VectorIndex,
        *,
        top_k: int = DEFAULT_TOP_K,
        candidates: int = DEFAULT_CANDIDATES,
        rrf_k: int = RRF_K,
    ):
        self.index = index
        self.top_k = max(1, top_k)
        self.candidates = max(self.top_k, candidates)
        self.rrf_k = rrf_k

    def lookup(
        self,
        queries: list[str],
        *,
        run_id: str,
        allowed_source_ids: set[str] | None,
    ) -> LookupResult:
        queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
        result = LookupResult()
        if not queries:
            return result
        with traced(
            "passage.lookup",
            as_type="retriever",
            metadata={"queries": len(queries), "top_k": self.top_k},
        ) as observation:
            rankings: list[tuple[str, list[IndexedChunk]]] = []
            by_id: dict[str, IndexedChunk] = {}
            vector_ok = True
            try:
                for query in queries:
                    rankings.append(
                        ("vector", self.index.retrieve(query, self.candidates, run_id=run_id))
                    )
            except Exception as exc:
                vector_ok = False
                result.error = redact_error(exc)
            try:
                documents = self.index.documents(run_id)
            except Exception as exc:
                documents = None
                result.fallback = "vector_only"
                if not vector_ok:
                    result.error = redact_error(exc)
            if documents:
                bm25 = BM25([chunk.text for chunk in documents])
                for query in queries:
                    rankings.append(
                        ("keyword", [documents[i] for i in bm25.top(query, self.candidates)])
                    )
            fused: dict[str, float] = {}
            best: dict[str, dict[str, int]] = {}
            for kind, chunks in rankings:
                rank = 0
                for chunk in chunks:
                    if allowed_source_ids is not None and chunk.source_id not in allowed_source_ids:
                        continue
                    rank += 1
                    by_id.setdefault(chunk.chunk_id, chunk)
                    fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
                    seen = best.setdefault(chunk.chunk_id, {})
                    seen[kind] = min(seen.get(kind, rank), rank)
            order = sorted(
                fused,
                key=lambda cid: (
                    -fused[cid],
                    best[cid].get("vector", 10**6),
                    cid,
                ),
            )[: self.top_k]
            result.chunks = [by_id[cid] for cid in order]
            result.ranks = [
                {
                    "chunk_id": cid,
                    "source_id": by_id[cid].source_id,
                    "rank": position,
                    "vector_rank": best[cid].get("vector"),
                    "keyword_rank": best[cid].get("keyword"),
                    "fused": round(fused[cid], 6),
                }
                for position, cid in enumerate(order, start=1)
            ]
            observation.output = {"hits": len(order), "fallback": result.fallback}
        return result
