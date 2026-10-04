"""Chroma-backed vector index: real embeddings for semantic retrieval, persisted locally.

Same adapter interface as :class:`LocalVectorIndex` (``index(chunks)`` / ``retrieve(query, ...)``,
see :class:`~intelligence_collector.storage.vector_index.VectorIndex`), so it is a drop-in
replacement wherever a vector index is injected. Chroma itself runs fully locally against a
directory on disk — no network call, no service to run. Only computing embeddings calls out
(to Google Gemini by default, OpenAI when explicitly selected); everything else, including
similarity search, happens locally.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from intelligence_collector.contracts import IndexedChunk
from intelligence_collector.observability.tracing import traced

DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_PROVIDER = "google_genai"
DEFAULT_COLLECTION = "research_chunks"


def collection_for(provider: str, model: str) -> str:
    """One collection per embedding provider and model. A collection holds vectors of a single
    dimension: reusing one name after switching models (observed: gemini-embedding-2, 3072
    dimensions -> text-embedding-3-small, 1536) made every upsert fail."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{provider}-{model}").strip("._-")
    return f"{DEFAULT_COLLECTION}_{slug}"[:512]


class Embeddings(Protocol):
    """The slice of ``langchain_core.embeddings.Embeddings`` this adapter needs.

    Real usage passes a ``GoogleGenerativeAIEmbeddings`` (or ``OpenAIEmbeddings``) instance
    (or nothing, to construct one); tests inject a deterministic fake so the adapter's own
    logic is covered offline.
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


def default_embeddings(model: str, provider: str) -> Embeddings:
    """The LangChain embeddings for `provider` ("openai" or Google Gemini otherwise); shared by
    the vector index and the RAGAS Response Relevancy metric."""
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings  # lazy: only needed for this provider

        return OpenAIEmbeddings(model=model)

    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # lazy: live default

    return GoogleGenerativeAIEmbeddings(model=model)


class ChromaVectorIndex:
    """Local, persistent Chroma collection embedded with a Google Gemini (or injected) model."""

    def __init__(
        self,
        path: str | Path,
        *,
        embeddings: Embeddings | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        provider: str = DEFAULT_EMBEDDING_PROVIDER,
        collection_name: str | None = None,
    ):
        import chromadb  # lazy: keeps the dependency edge local to this adapter

        Path(path).mkdir(parents=True, exist_ok=True)
        self._embeddings = embeddings or default_embeddings(model, provider)
        self._embedding_model = f"{provider}/{model}"
        self._collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
            collection_name or collection_for(provider, model)
        )

    def index(self, chunks: list[IndexedChunk]) -> list[str]:
        """Embed and upsert the chunks; returns their ids."""
        if not chunks:
            return []
        with traced(
            "vector_index.upsert",
            as_type="embedding",
            metadata={"embedding_model": self._embedding_model, "chunks": len(chunks)},
        ):
            self._upsert(chunks)
        return [chunk.chunk_id for chunk in chunks]

    def _upsert(self, chunks: list[IndexedChunk]) -> None:
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

    def retrieve(
        self,
        query: str,
        limit: int = 8,
        *,
        run_id: str | None = None,
        todo_id: str | None = None,
    ) -> list[IndexedChunk]:
        """Nearest chunks to `query`, optionally scoped to one run and one research question."""
        if self._collection.count() == 0:
            return []
        with traced(
            "vector_index.query",
            as_type="retriever",
            input={"query": query},
            metadata={"todo_id": todo_id, "limit": limit, "embedding_model": self._embedding_model},
        ) as observation:
            result = self._collection.query(
                query_embeddings=[self._embeddings.embed_query(query)],
                n_results=limit,
                where=_where(run_id, todo_id),
            )
            ids = result.get("ids") or [[]]
            observation.output = {"hits": len(ids[0])}
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
