import hashlib

from intelligence_collector.contracts import IndexedChunk


class FakeVectorIndex:
    """In-memory vector-index substitute preserving chunk provenance."""
    def __init__(self):
        self.chunks = []

    def index(self, chunks):
        self.chunks.extend(chunks)
        return [chunk.chunk_id for chunk in chunks]

    def retrieve(self, query, limit=8, *, run_id=None, todo_id=None):
        rows = self.chunks
        if run_id is not None:
            rows = [c for c in rows if c.metadata.get("run_id") == run_id]
        if todo_id is not None:
            rows = [c for c in rows if c.metadata.get("todo_id") == todo_id]
        return list(rows[:limit])


class FakeEmbeddings:
    """Deterministic bag-of-words embedding: no network, no API key, offline-testable.

    Shared words between two texts pull their vectors closer together, so retrieval ranking
    is meaningful in tests without depending on a real embedding model.
    """

    def __init__(self, dims: int = 32):
        self.dims = dims
        self.calls: list[str] = []

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        for word in text.lower().split():
            bucket = int(hashlib.sha1(word.encode()).hexdigest(), 16) % self.dims
            vector[bucket] += 1.0
        norm = sum(v * v for v in vector) ** 0.5
        return [v / norm for v in vector] if norm else vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._vector(text)

