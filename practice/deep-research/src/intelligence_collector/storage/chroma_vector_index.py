"""Chroma-backed vector index: real embeddings for semantic retrieval, persisted locally.

Same adapter interface as :class:`LocalVectorIndex` (``index(chunks)`` / ``retrieve(query, ...)``,
see :class:`~intelligence_collector.storage.vector_index.VectorIndex`), so it is a drop-in
replacement wherever a vector index is injected. Chroma itself runs fully locally against a
directory on disk — no network call, no service to run. Only computing embeddings calls out
(to OpenAI by default); everything else, including similarity search, happens locally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from intelligence_collector.contracts import IndexedChunk

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_COLLECTION = "research_chunks"


class Embeddings(Protocol):
    """The slice of ``langchain_core.embeddings.Embeddings`` this adapter needs.

    Real usage passes an ``OpenAIEmbeddings`` instance (or nothing, to construct one);
    tests inject a deterministic fake so the adapter's own logic is covered offline.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def _where(run_id: str | None, todo_id: str | None) -> dict | None:
    clauses = [{"run_id": run_id}] if run_id else []
    if todo_id:
        clauses.append({"todo_id": todo_id})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _default_embeddings(model: str) -> Embeddings:
    from langchain_openai import OpenAIEmbeddings  # lazy: only needed for the live default

    return OpenAIEmbeddings(model=model)


class ChromaVectorIndex:
    """Local, persistent Chroma collection embedded with an OpenAI (or injected) embedding model."""

    def __init__(
        self,
        path: str | Path,
        *,
        embeddings: Embeddings | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        collection_name: str = DEFAULT_COLLECTION,
    ):
        import chromadb  # lazy: keeps the dependency edge local to this adapter

        Path(path).mkdir(parents=True, exist_ok=True)
        self._embeddings = embeddings or _default_embeddings(model)
        self._collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
            collection_name
        )

    def index(self, chunks: list[IndexedChunk]) -> list[str]:
        if not chunks:
            return []
        vectors = self._embeddings.embed_documents([chunk.text for chunk in chunks])
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "run_id": str(chunk.metadata.get("run_id", "")),
                    "todo_id": str(chunk.metadata.get("todo_id", "")),
                    "locator": str(chunk.metadata.get("locator", "")),
                    "source_id": chunk.source_id,
                    "artifact_id": chunk.artifact_id,
                }
                for chunk in chunks
            ],
        )
        return [chunk.chunk_id for chunk in chunks]

    def retrieve(
        self,
        query: str,
        limit: int = 8,
        *,
        run_id: str | None = None,
        todo_id: str | None = None,
    ) -> list[IndexedChunk]:
        if self._collection.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[self._embeddings.embed_query(query)],
            n_results=limit,
            where=_where(run_id, todo_id),
        )
        ids = result.get("ids") or [[]]
        documents = result.get("documents") or [[]]
        metadatas = result.get("metadatas") or [[]]
        return [
            IndexedChunk(
                chunk_id=chunk_id,
                source_id=metadata.get("source_id", ""),
                artifact_id=metadata.get("artifact_id", ""),
                text=text,
                metadata={
                    "run_id": metadata.get("run_id", ""),
                    "todo_id": metadata.get("todo_id", ""),
                    "locator": metadata.get("locator", ""),
                },
            )
            for chunk_id, text, metadata in zip(ids[0], documents[0], metadatas[0])
        ]
