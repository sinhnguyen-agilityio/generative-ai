from __future__ import annotations
import json
from pathlib import Path
from intelligence_collector.contracts import IndexedChunk


class LocalVectorIndex:
    """Deterministic lexical fallback; adapter boundary can be replaced by an embedding index."""

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
        self, query: str, limit: int = 8, *, run_id: str | None = None
    ) -> list[IndexedChunk]:
        terms = set(query.lower().split())
        rows = self._all()
        if run_id is not None:
            rows = [row for row in rows if row.get(
                "metadata", {}).get("run_id") == run_id]
        rows.sort(key=lambda r: len(terms & set(
            r["text"].lower().split())), reverse=True)
        return [IndexedChunk.model_validate(r) for r in rows[:limit]]
