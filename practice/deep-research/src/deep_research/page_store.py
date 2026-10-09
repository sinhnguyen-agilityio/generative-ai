"""Page store definitions."""

from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from deep_research.utils.embeddings import _embeddings

MAX_CHUNKS_PER_URL = 2  # so one page cannot fill all retrieved passages


class PageStore:
    """Result pages in a local Chroma collection, chunked and embedded, searchable per run."""

    def __init__(self, path: str, model: str, provider: str):
        """Open a persistent Chroma collection and configure embeddings and page chunking.

        Args:
            path: Directory for the local Chroma database.
            model: Embedding model determining the collection and vector dimensions.
            provider: Embedding provider used to create the client and collection name.
        """
        import chromadb

        Path(path).mkdir(parents=True, exist_ok=True)
        self._embeddings = _embeddings(model, provider)
        # One collection per embedding model: vectors of different sizes cannot share one.
        name = f"pages_cos_{provider}_{model}".replace("/", "-")[:60]
        self._collection = chromadb.PersistentClient(path=path).get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100)

    def add(self, pages: list[dict], *, run_id: str, topic: str) -> None:
        """Chunk and embed pages, then upsert them into the current run.

        Chunk IDs depend on run, URL, and chunk index, so repeated pages replace existing
        chunks with matching IDs even when another topic discovers them.

        Args:
            pages: Prepared pages including text and source metadata.
            run_id: Research thread used to isolate indexed evidence.
            topic: Delegated topic recorded in chunk metadata.
        """
        ids, texts, metadatas = [], [], []
        for page in pages:
            for i, chunk in enumerate(self._splitter.split_text(page["text"])):
                # No topic in the id: a page has one set of chunks per run, whichever topic found it.
                ids.append(hashlib.sha1(
                    f"{run_id}|{page['url']}|{i}".encode()).hexdigest())
                texts.append(chunk)
                metadatas.append({
                    "run_id": run_id, "topic": topic, "url": page["url"], "title": page["title"],
                    "full_text": page["full_text"], "score": page["score"],
                })
        if ids:
            vectors = self._embeddings.embed_documents(texts)
            self._collection.upsert(
                ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    def query(self, text: str, *, run_id: str, urls: list[str], k: int) -> list[dict]:
        """Retrieve similar passages from the requested pages of one run.

        Args:
            text: Query embedded for similarity search.
            run_id: Thread identifier restricting the collection search.
            urls: Page URLs eligible for retrieval.
            k: Maximum returned passages, also subject to the per-URL chunk cap.

        Returns:
            Up to k URL/text passages in retrieval order.
        """
        result = self._collection.query(
            query_embeddings=[self._embeddings.embed_query(text)],
            n_results=k * MAX_CHUNKS_PER_URL,
            where={"$and": [{"run_id": run_id}, {"url": {"$in": urls}}]},
        )
        passages, per_url = [], {}
        for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
            if per_url.get(meta["url"], 0) < MAX_CHUNKS_PER_URL:
                per_url[meta["url"]] = per_url.get(meta["url"], 0) + 1
                passages.append({"url": meta["url"], "text": doc})
        return passages[:k]
