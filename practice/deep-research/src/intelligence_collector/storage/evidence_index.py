"""Vector-index records built from the VFS, the source of truth for collected pages.

Each captured page is split into chunks (``tools/page_content.chunk_page``) and every chunk is
embedded with the page title as a header, so a passage keeps its context. Chunks are derived
data: they are never stored on their own and can be rebuilt from the VFS whenever the index
changes — for example after switching the embedding model, which uses a separate collection.

A run records which index already holds its chunks (``index/<identity>.json``); a different
index identity triggers one re-index of the run's stored pages before retrieval.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from intelligence_collector.ports import ArtifactStore, VectorIndex
from intelligence_collector.contracts import IndexedChunk
from intelligence_collector.tools.page_content import chunk_page, source_id_for


def chunk_texts(title: str, text: str) -> list[str]:
    """The text of each chunk of a page as it is embedded: the page title as a header, then the
    chunk. Passage selection embeds exactly these texts, so the index can reuse their vectors."""
    header = f"{title}\n\n" if title else ""
    return [header + chunk for chunk in chunk_page(text)]


def source_chunks(
    *,
    run_id: str,
    todo_id: str,
    source_id: str,
    artifact_id: str,
    locator: str,
    title: str,
    text: str,
) -> list[IndexedChunk]:
    """The index records of one page. Chunk ids are stable per run and page, so indexing the
    same page again replaces its records instead of duplicating them."""
    return [
        IndexedChunk(
            chunk_id=f"{run_id}:{source_id}:{position}",
            source_id=source_id,
            artifact_id=artifact_id,
            text=chunk,
            metadata={
                "run_id": run_id,
                "todo_id": todo_id,
                "locator": locator,
                "chunk_index": position,
            },
        )
        for position, chunk in enumerate(chunk_texts(title, text))
    ]


def index_identity(index: VectorIndex | None) -> str | None:
    """Which collection an index writes to; None for an index that cannot say."""
    identity = getattr(index, "identity", None)
    return str(identity) if identity else None


def _marker_path(identity: str) -> str:
    return f"index/{re.sub(r'[^A-Za-z0-9._-]+', '-', identity)[:120]}.json"


def mark_indexed(vfs: ArtifactStore, run_id: str, index: VectorIndex) -> None:
    """Record that `index` holds this run's chunks (written whenever chunks are indexed)."""
    identity = index_identity(index)
    if identity is not None:
        vfs.write_json(run_id, _marker_path(identity), {"identity": identity}, "index_marker")


def stored_source_text(vfs: ArtifactStore, run_id: str, source: dict) -> str:
    """A captured page's full text from the VFS, or its excerpt when no page was stored."""
    raw_path = source.get("raw_path")
    if raw_path:
        try:
            return vfs.read(run_id, raw_path).decode("utf-8")
        except FileNotFoundError:
            pass
    return str(source.get("excerpt") or "")


def stored_text_for_source(vfs: ArtifactStore, run_id: str, source: dict) -> str:
    """The text a quote from `source` is checked against: its stored cleaned page, plus the
    passages selected from it (or only its snippet when no page was stored)."""
    page = stored_source_text(vfs, run_id, source)
    excerpt = str(source.get("excerpt") or source.get("content") or "")
    return page if excerpt and excerpt in page else "\n".join(t for t in (page, excerpt) if t)


def reindex_run(vfs: ArtifactStore, index: VectorIndex, run_id: str) -> int:
    """Rebuild every chunk of the run from its stored evidence bundles; returns the count.

    Sources a bundle only references (``reused_from_task``) are indexed by the task that
    captured them.
    """
    base = Path(vfs.root) / run_id
    chunks: list[IndexedChunk] = []
    for path in sorted(base.glob("research/**/evidence/result.json")):
        bundle = json.loads(path.read_text(encoding="utf-8"))
        for source in bundle.get("sources", []):
            if source.get("reused_from_task"):
                continue
            locator = str(source.get("locator", ""))
            chunks.extend(
                source_chunks(
                    run_id=run_id,
                    todo_id=str(bundle.get("task_id", "")),
                    source_id=str(source.get("source_id") or source_id_for(locator)),
                    artifact_id=str(source.get("artifact_id", "")),
                    locator=locator,
                    title=str((source.get("provenance") or {}).get("title", "")),
                    text=stored_source_text(vfs, run_id, source),
                )
            )
    if chunks:
        index.index(chunks)
    return len(chunks)


def ensure_run_indexed(vfs: ArtifactStore, index: VectorIndex, run_id: str) -> int:
    """Re-index the run once when `index` has never received its chunks (e.g. a new embedding
    model). Returns the number of chunks indexed, 0 when nothing was needed."""
    identity = index_identity(index)
    if index is None or identity is None:
        return 0
    try:
        vfs.read(run_id, _marker_path(identity))
        return 0
    except FileNotFoundError:
        pass
    count = reindex_run(vfs, index, run_id)
    mark_indexed(vfs, run_id, index)
    return count
