"""Local MCP server for approved embedding providers and persistent Chroma.

Run: python -m intelligence_collector.storage.mcp_server
The client owns retries; this server makes one provider attempt per tool invocation.
Google text embeddings use LangChain with the configured model and provider-default
dimensions/task type, preserving compatibility with existing stored vectors.
"""
from __future__ import annotations
import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from intelligence_collector.config import load_local_env

load_local_env()

server = FastMCP("research-storage", host="127.0.0.1",
                 port=int(os.getenv("INTELLIGENCE_STORAGE_MCP_PORT", "8000")))


@server.tool()
def embed(texts: list[str], model: str, provider: str) -> dict:
    from intelligence_collector.models.router import _is_transient
    from intelligence_collector.models.redaction import redact_text
    try:
        return _embed(texts, model, provider)
    except Exception as exc:
        return {"error": {"retryable": _is_transient(exc), "type": type(exc).__name__,
                          "message": redact_text(exc),
                          "status_code": getattr(exc, "status_code", getattr(exc, "code", None))}}


def _embed(texts: list[str], model: str, provider: str) -> dict:
    if not texts or any(not text.strip() for text in texts):
        raise ValueError("Embedding requires non-empty texts")
    if provider == "openai":
        from openai import OpenAI
        result = OpenAI(max_retries=0, timeout=60).embeddings.create(model=model, input=texts)
        vectors = [entry.embedding for entry in sorted(result.data, key=lambda e: e.index)]
    elif provider == "google_genai":
        from google.genai.types import HttpOptions
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_google_genai._common import GoogleGenerativeAIError

        class StorageEmbeddings(GoogleGenerativeAIEmbeddings):
            def _build_config(self, **kwargs):
                config = super()._build_config(**kwargs)
                # Retry ownership remains with the durable MCP client. Configure
                # per-request options because LangChain's request_options field
                # is not applied by the installed integration.
                config.http_options = HttpOptions(timeout=60000, retry_options={"attempts": 1})
                return config

        embeddings = StorageEmbeddings(
            model=model, task_type="TASK_TYPE_UNSPECIFIED", output_dimensionality=None,
        )
        # Preserve the old provider-default task and dimensions for existing
        # indexes. LangChain keeps each text as a separate Content and batches it.
        try:
            with embeddings.client:
                vectors = embeddings.embed_documents(texts)
        except GoogleGenerativeAIError as exc:
            # LangChain wraps SDK errors; retain status codes for the MCP error
            # envelope and transient-failure classification.
            if exc.__cause__ is not None:
                raise exc.__cause__ from exc
            raise
    else:
        raise ValueError("Unapproved embedding provider")
    if len(vectors) != len(texts):
        raise ValueError(f"Partial embeddings: expected {len(texts)}, received {len(vectors)}")
    return {"embeddings": vectors}


def collection(model: str = "", provider: str = ""):
    import hashlib
    import chromadb
    target = Path(os.getenv("INTELLIGENCE_CHROMA_PATH", ".data/chroma"))
    target.mkdir(parents=True, exist_ok=True)
    name = "research_chunks"
    if model or provider:
        if not model or not provider:
            raise ValueError("Collection requires both embedding model and provider")
        identity = hashlib.sha256(f"{provider}:{model}".encode()).hexdigest()[:24]
        name = f"research_chunks_{identity}_v3"
    return chromadb.PersistentClient(path=str(target)).get_or_create_collection(name)


@server.tool()
def chroma_upsert(run_id: str, chunks: list[dict], embeddings: list[list[float]],
                  model: str = "", provider: str = "") -> dict:
    from intelligence_collector.models.router import _is_transient
    from intelligence_collector.models.redaction import redact_text
    try:
        return _chroma_upsert(run_id, chunks, embeddings, model, provider)
    except Exception as exc:
        return {"error": {"retryable": _is_transient(exc), "type": type(exc).__name__,
                          "message": redact_text(exc),
                          "status_code": getattr(exc, "status_code", None)}}


def _chroma_upsert(run_id, chunks, embeddings, model, provider):
    from intelligence_collector.contracts import IndexedChunk
    parsed = [IndexedChunk.model_validate(c) for c in chunks]
    if len(parsed) != len(embeddings) or any(c.metadata.get("run_id") != run_id for c in parsed):
        raise ValueError("Index identity or embedding count mismatch")
    metadata = [{**{k: str(v) for k, v in c.metadata.items()},
                 "source_id": c.source_id, "artifact_id": c.artifact_id} for c in parsed]
    collection(model, provider).upsert(ids=[c.chunk_id for c in parsed], embeddings=embeddings,
                        documents=[c.text for c in parsed], metadatas=metadata)
    return {"chunk_ids": [c.chunk_id for c in parsed]}


@server.tool()
def chroma_query(run_id: str, embedding: list[float], query: str = "",
                 todo_id: str | None = None, limit: int = 8, model: str = "",
                 provider: str = "", evidence_version: str = "", index_identity: str = "") -> dict:
    store = collection(model, provider)
    if not run_id:
        raise ValueError("Retrieval requires a run ID")
    if not store.count():
        return {"chunks": []}
    where = {"run_id": run_id} if not todo_id else {
        "$and": [{"run_id": run_id}, {"todo_id": todo_id}]}
    result = store.query(query_embeddings=[embedding], n_results=min(max(1, limit), store.count()),
                         where=where)
    chunks = []
    for ident, text, meta in zip(result["ids"][0], result["documents"][0], result["metadatas"][0]):
        chunks.append({"chunk_id": ident, "source_id": meta.pop("source_id"),
                       "artifact_id": meta.pop("artifact_id"), "text": text, "metadata": meta})
    return {"chunks": chunks}


if __name__ == "__main__":
    server.run(transport="streamable-http")
