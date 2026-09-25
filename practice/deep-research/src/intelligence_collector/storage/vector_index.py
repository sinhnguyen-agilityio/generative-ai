from __future__ import annotations
import json
from pathlib import Path
from typing import Protocol

from intelligence_collector.contracts import IndexedChunk


class VectorIndex(Protocol):
    """Adapter boundary implemented by every vector-store backend (local lexical, Chroma, ...).

    ``retrieve`` scoping: ``run_id`` scopes to one run; ``todo_id`` further scopes to one research
    question within that run, which lets a caller ask "the most relevant chunks *for this task*"
    without results from a sibling task crowding them out.
    """

    def index(self, chunks: list[IndexedChunk]) -> list[str]: ...

    def retrieve(
        self,
        query: str,
        limit: int = 8,
        *,
        run_id: str | None = None,
        todo_id: str | None = None,
    ) -> list[IndexedChunk]: ...


class LocalVectorIndex:
    """Deterministic lexical fallback; no external service or API key required.

    Ranks by literal word overlap, not semantic similarity. Good enough for tests and small
    offline runs; swap in ``ChromaVectorIndex`` (real embeddings) for meaningful ranking.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _all(self) -> list[dict]: return json.loads(
        self.path.read_text()) if self.path.exists() else []

    def index(self, chunks: list[IndexedChunk]) -> list[str]:
        rows = self._all()
        rows.extend([c.model_dump(mode="json") for c in chunks])
        self.path.write_text(json.dumps(rows), encoding="utf-8")
        return [c.chunk_id for c in chunks]

    def retrieve(
        self,
        query: str,
        limit: int = 8,
        *,
        run_id: str | None = None,
        todo_id: str | None = None,
    ) -> list[IndexedChunk]:
        terms = set(query.lower().split())
        rows = self._all()
        if run_id is not None:
            rows = [row for row in rows if row.get(
                "metadata", {}).get("run_id") == run_id]
        if todo_id is not None:
            rows = [row for row in rows if row.get(
                "metadata", {}).get("todo_id") == todo_id]
        rows.sort(key=lambda r: len(terms & set(
            r["text"].lower().split())), reverse=True)
        return [IndexedChunk.model_validate(r) for r in rows[:limit]]
