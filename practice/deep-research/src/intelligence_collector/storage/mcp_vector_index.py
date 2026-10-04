"""Embedding and Chroma ports over the configured MCP server only."""

from __future__ import annotations
from datetime import timedelta
import json
from intelligence_collector.tools.async_bridge import run_async
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from intelligence_collector.contracts import IndexedChunk
from intelligence_collector.services.workflow import DurableOperations, digest


class MCPStorageTransport:
    def __init__(self, endpoint, *, timeout_seconds=60):
        from urllib.parse import urlsplit

        if urlsplit(endpoint).scheme not in {"https", "http"}:
            raise ValueError("Configure INTELLIGENCE_STORAGE_MCP_URL")
        self.endpoint, self.timeout_seconds = endpoint, timeout_seconds

    async def _call(self, tool, arguments):
        async with streamable_http_client(self.endpoint) as streams:
            async with ClientSession(
                *streams[:2], read_timeout_seconds=timedelta(seconds=self.timeout_seconds)
            ) as session:
                await session.initialize()
                result = await session.call_tool(
                    tool, arguments, read_timeout_seconds=timedelta(seconds=self.timeout_seconds)
                )
                if result.isError:
                    from intelligence_collector.models.redaction import redact_text

                    detail = redact_text(
                        " ".join(b.text for b in result.content if b.type == "text")
                    )
                    raise RuntimeError(f"MCP storage tool {tool} failed: {detail}")
                payload = getattr(result, "structured_content", None)
                if payload is None:
                    payload = json.loads(
                        "".join(b.text for b in result.content if b.type == "text")
                    )
                if isinstance(payload, dict) and payload.get("error"):
                    error = payload["error"]
                    from intelligence_collector.models.redaction import redact_text

                    detail = redact_text(
                        f"{error.get('type', 'UnknownError')} "
                        f"(status={error.get('status_code')}): {error.get('message', '')}"
                    )
                    if error.get("retryable"):
                        raise ConnectionError(f"Transient MCP storage failure: {tool}: {detail}")
                    raise ValueError(f"Non-retryable MCP storage failure: {tool}: {detail}")
                return payload

    def call(self, tool, arguments):
        return run_async(self._call, tool, arguments)


class MCPVectorIndex:
    """Server contract: embed, chroma_upsert, chroma_query; VFS owns saved embeddings."""

    def __init__(self, transport, vfs, *, model, provider):
        self.transport, self.vfs = transport, vfs
        self.model, self.provider = model, provider
        self.operations = DurableOperations(vfs)
        self.identity = digest({"model": model, "provider": provider, "storage_contract": 3})

    def _split_chunks(self, chunks):
        if self.provider != "openai":
            return list(chunks)
        import tiktoken

        encoding = tiktoken.encoding_for_model(self.model)

        def split(text):
            if len(encoding.encode(text, disallowed_special=())) <= 2000:
                return [text]
            middle = len(text) // 2
            return split(text[:middle]) + split(text[middle:])

        result = []
        for chunk in chunks:
            parts = split(chunk.text)
            if len(parts) == 1:
                result.append(chunk)
                continue
            for position, text in enumerate(parts):
                result.append(
                    chunk.model_copy(
                        update={
                            "chunk_id": f"{chunk.chunk_id}:part:{position}",
                            "text": text,
                            "metadata": {
                                **chunk.metadata,
                                "parent_chunk_id": chunk.chunk_id,
                                "chunk_position": position,
                            },
                        }
                    )
                )
        return result

    def index(self, chunks):
        if not chunks:
            return []
        chunks = self._split_chunks(chunks)
        grouped = {}
        for chunk in chunks:
            grouped.setdefault(chunk.metadata["run_id"], []).append(chunk)
        for run_id, run_chunks in grouped.items():
            # Bound total request tokens as well as each individual input.
            for start in range(0, len(run_chunks), 64):
                self._index_batch(run_id, run_chunks[start : start + 64])
        return [c.chunk_id for c in chunks]

    def _index_batch(self, run_id, group):
        rows = [c.model_dump(mode="json") for c in group]
        inputs = {
            "storage_contract": 3,
            "rows": rows,
            "model": self.model,
            "provider": self.provider,
        }
        vectors = self.operations.call(
            run_id,
            "embedding",
            inputs,
            lambda: self.transport.call(
                "embed",
                {"texts": [c.text for c in group], "model": self.model, "provider": self.provider},
            ),
        )
        if not isinstance(vectors, dict) or len(vectors.get("embeddings", [])) != len(group):
            raise ValueError("MCP embeddings returned partial results")
        upsert = self.operations.call(
            run_id,
            "chroma_upsert",
            {**inputs, "index_identity": self.identity, "upsert_contract": 4},
            lambda: self.transport.call(
                "chroma_upsert",
                {
                    "run_id": run_id,
                    "chunks": rows,
                    "embeddings": vectors["embeddings"],
                    "model": self.model,
                    "provider": self.provider,
                },
            ),
        )
        if sorted(upsert.get("chunk_ids", [])) != sorted(c.chunk_id for c in group):
            raise ValueError("Partial Chroma upsert")
        try:
            revision = json.loads(self.vfs.read(run_id, "storage/index_version.json"))
        except FileNotFoundError:
            revision = {}
        revision.update({c.chunk_id: digest(c.model_dump(mode="json")) for c in group})
        self.vfs.write_json(run_id, "storage/index_version.json", revision, "index_version")

    def retrieve(self, query, limit=8, *, run_id=None, todo_id=None):
        if not run_id:
            raise ValueError("MCP retrieval requires a run ID")
        try:
            revision = json.loads(self.vfs.read(run_id, "storage/index_version.json"))
        except FileNotFoundError:
            revision = {}
        inputs = {
            "index_identity": self.identity,
            "evidence_version": digest(revision),
            "query": query,
            "run_id": run_id,
            "todo_id": todo_id,
            "limit": limit,
            "model": self.model,
            "provider": self.provider,
        }

        def query_index():
            vectors = self.operations.call(
                run_id,
                "query_embedding",
                inputs,
                lambda: self.transport.call(
                    "embed", {"texts": [query], "model": self.model, "provider": self.provider}
                ),
            )
            return self.transport.call(
                "chroma_query", {**inputs, "embedding": vectors["embeddings"][0]}
            )

        # An evidence version is part of callers' queries when freshness matters.
        payload = self.operations.call(run_id, "chroma_query", inputs, query_index)
        return [IndexedChunk.model_validate(row) for row in payload["chunks"]]


class MCPEmbeddings:
    """LangChain-shaped embeddings port for optional RAGAS metrics, still MCP-only."""

    def __init__(self, index):
        self.index = index

    def embed_documents(self, texts):
        from intelligence_collector.services.operation_context import operation_run

        run_id = operation_run.get()
        if not run_id:
            raise ValueError("Embedding requires an active research run")
        payload = self.index.operations.call(
            run_id,
            "metric_embedding",
            {"texts": texts, "model": self.index.model, "provider": self.index.provider},
            lambda: self.index.transport.call(
                "embed",
                {"texts": texts, "model": self.index.model, "provider": self.index.provider},
            ),
        )
        return payload["embeddings"]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts):
        import asyncio

        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text):
        return (await self.aembed_documents([text]))[0]
