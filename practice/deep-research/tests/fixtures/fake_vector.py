from intelligence_collector.contracts import IndexedChunk


class FakeVectorIndex:
    """In-memory vector-index substitute preserving chunk provenance."""
    def __init__(self):
        self.chunks = []

    def index(self, chunks):
        self.chunks.extend(chunks)
        return [chunk.chunk_id for chunk in chunks]

    def retrieve(self, query, limit=8):
        return list(self.chunks[:limit])

