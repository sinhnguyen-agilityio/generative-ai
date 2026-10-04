from intelligence_collector.contracts import IndexedChunk
from intelligence_collector.storage.chroma_vector_index import ChromaVectorIndex
from intelligence_collector.storage.vector_index import LocalVectorIndex
from tests.fixtures.fake_vector import FakeEmbeddings


def _chunk(text, *, run_id="r1", todo_id="t1", locator="https://x.test", source_id="s1"):
    return IndexedChunk(
        source_id=source_id,
        artifact_id="a1",
        text=text,
        metadata={"run_id": run_id, "todo_id": todo_id, "locator": locator},
    )


# ===== LocalVectorIndex: todo_id scoping (added alongside the Chroma adapter) =====


def test_local_index_retrieve_scopes_by_run_and_todo_id(tmp_path):
    index = LocalVectorIndex(tmp_path / "index.json")
    index.index([
        _chunk("cache eviction policy", run_id="r1", todo_id="t1"),
        _chunk("cache eviction policy", run_id="r1", todo_id="t2"),
        _chunk("cache eviction policy", run_id="r2", todo_id="t1"),
    ])

    by_run = index.retrieve("cache eviction policy", run_id="r1")
    by_task = index.retrieve("cache eviction policy", run_id="r1", todo_id="t2")

    assert {c.metadata["run_id"] for c in by_run} == {"r1"}
    assert len(by_run) == 2
    assert len(by_task) == 1 and by_task[0].metadata["todo_id"] == "t2"


# ===== ChromaVectorIndex: real vector-store adapter, embeddings faked for offline tests =====


def test_chroma_index_ranks_by_similarity_not_insertion_order(tmp_path):
    index = ChromaVectorIndex(tmp_path / "chroma", embeddings=FakeEmbeddings())
    index.index([
        _chunk("The weather today is sunny and warm", source_id="off-topic"),
        _chunk("Caches evict the least recently used entry first", source_id="on-topic"),
    ])

    results = index.retrieve("How do caches evict entries?", limit=1)

    assert [r.source_id for r in results] == ["on-topic"]


def test_chroma_index_scopes_by_run_id_and_todo_id(tmp_path):
    index = ChromaVectorIndex(tmp_path / "chroma", embeddings=FakeEmbeddings())
    index.index([
        _chunk("cache eviction details", run_id="r1", todo_id="t1", source_id="mine"),
        _chunk("cache eviction details", run_id="r1", todo_id="t2", source_id="sibling"),
        _chunk("cache eviction details", run_id="r2", todo_id="t1", source_id="other-run"),
    ])

    scoped = index.retrieve("cache eviction details", limit=10, run_id="r1", todo_id="t1")

    assert [r.source_id for r in scoped] == ["mine"]


def test_chroma_index_round_trips_provenance_metadata(tmp_path):
    index = ChromaVectorIndex(tmp_path / "chroma", embeddings=FakeEmbeddings())
    index.index([_chunk(
        "Documented locking behavior", locator="https://docs.test/locking", source_id="S3",
    )])

    [result] = index.retrieve("locking behavior")

    assert result.source_id == "S3"
    assert result.artifact_id == "a1"
    assert result.metadata["locator"] == "https://docs.test/locking"
    assert result.text == "Documented locking behavior"


def test_chroma_index_empty_collection_returns_no_results(tmp_path):
    index = ChromaVectorIndex(tmp_path / "chroma", embeddings=FakeEmbeddings())

    assert index.retrieve("anything") == []
    assert index.index([]) == []


def test_chroma_index_persists_across_instances(tmp_path):
    ChromaVectorIndex(tmp_path / "chroma", embeddings=FakeEmbeddings()).index([
        _chunk("persisted cache content", source_id="durable"),
    ])

    reopened = ChromaVectorIndex(tmp_path / "chroma", embeddings=FakeEmbeddings())
    [result] = reopened.retrieve("persisted cache content")

    assert result.source_id == "durable"


def test_chroma_index_uses_google_embeddings_by_default(tmp_path, monkeypatch):
    captured = {}

    class RecordingGoogleEmbeddings:
        def __init__(self, model):
            captured["model"] = model

    monkeypatch.setattr(
        "langchain_google_genai.GoogleGenerativeAIEmbeddings", RecordingGoogleEmbeddings,
    )

    ChromaVectorIndex(tmp_path / "chroma", model="gemini-embedding-2")

    assert captured["model"] == "gemini-embedding-2"


def test_chroma_index_uses_openai_embeddings_when_provider_is_openai(tmp_path, monkeypatch):
    captured = {}

    class RecordingOpenAIEmbeddings:
        def __init__(self, model):
            captured["model"] = model

    monkeypatch.setattr(
        "langchain_openai.OpenAIEmbeddings", RecordingOpenAIEmbeddings,
    )

    ChromaVectorIndex(
        tmp_path / "chroma", model="text-embedding-3-large", provider="openai",
    )

    assert captured["model"] == "text-embedding-3-large"


def test_switching_embedding_models_uses_a_separate_collection(tmp_path):
    """A 3072-dimension collection must not receive 1536-dimension vectors after a model switch."""
    class Dim:
        def __init__(self, size):
            self.size = size

        def embed_documents(self, texts):
            return [[0.1] * self.size for _ in texts]

        def embed_query(self, text):
            return [0.1] * self.size

    chunk = IndexedChunk(source_id="s", artifact_id="a", text="cache text",
                         metadata={"run_id": "r", "todo_id": "t", "locator": "https://x.test"})
    ChromaVectorIndex(tmp_path / "chroma", embeddings=Dim(8), model="gemini-embedding-2",
                      provider="google_genai").index([chunk])

    switched = ChromaVectorIndex(tmp_path / "chroma", embeddings=Dim(4), model="text-embedding-3-small",
                                 provider="openai")
    switched.index([chunk])

    assert [c.chunk_id for c in switched.retrieve("cache", run_id="r")] == [chunk.chunk_id]
