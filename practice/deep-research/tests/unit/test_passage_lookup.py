"""Hybrid stored-passage lookup (spec 002, FR-013d to FR-013f; contracts/passage-lookup.md)."""

from intelligence_collector.contracts import IndexedChunk
from intelligence_collector.storage.passage_lookup import BM25, PassageLookup


def chunk(cid, text, source="src_a", run="run"):
    return IndexedChunk(
        chunk_id=cid,
        source_id=source,
        artifact_id="art",
        text=text,
        metadata={"run_id": run, "locator": f"https://{source}.example"},
    )


CHUNKS = [
    chunk("c1", "SQLite is lightweight, small and self-contained."),
    chunk("c2", "SQLite is serverless and needs zero configuration."),
    chunk("c3", "The maximum database size is 281 TB (2^48 bytes).", source="src_b"),
    chunk("c4", "PostgreSQL uses multiversion concurrency control.", source="src_c"),
    chunk("c5", "Unrelated chunk of another run.", run="other"),
]


class FakeIndex:
    """Vector ranking scripted per query; documents() serves the run's chunks."""

    def __init__(self, vector_order, *, documents=True, vector_fails=False):
        self.vector_order = vector_order
        self.documents_ok = documents
        self.vector_fails = vector_fails
        self.writes = 0

    def index(self, chunks):
        self.writes += 1
        return []

    def retrieve(self, query, limit=8, *, run_id=None, todo_id=None):
        if self.vector_fails:
            raise RuntimeError("vector store down; api_key=sk-secret-123")
        by_id = {c.chunk_id: c for c in CHUNKS}
        hits = [by_id[cid] for cid in self.vector_order]
        return [c for c in hits if c.metadata["run_id"] == run_id][:limit]

    def documents(self, run_id):
        if not self.documents_ok:
            raise NotImplementedError("no documents API")
        return [c for c in CHUNKS if c.metadata["run_id"] == run_id]


def test_keyword_ranking_brings_an_exact_number_into_the_top_results():
    # Meaning-based ranking misses the size passage entirely ("small" ~ "size").
    index = FakeIndex(["c1", "c2", "c4"])
    lookup = PassageLookup(index, top_k=3)

    result = lookup.lookup(
        ["SQLite maximum database size 281 TB"], run_id="run", allowed_source_ids=None
    )

    assert "c3" in [c.chunk_id for c in result.chunks]
    rank = next(r for r in result.ranks if r["chunk_id"] == "c3")
    assert rank["keyword_rank"] == 1 and rank["vector_rank"] is None
    assert result.fallback == "none" and index.writes == 0


def test_allowed_sources_filter_and_other_runs_are_excluded():
    index = FakeIndex(["c1", "c3", "c4", "c5"])
    result = PassageLookup(index, top_k=5).lookup(
        ["database size"], run_id="run", allowed_source_ids={"src_a", "src_b"}
    )
    assert {c.source_id for c in result.chunks} <= {"src_a", "src_b"}
    assert "c5" not in [c.chunk_id for c in result.chunks]


def test_fusion_rewards_agreement_between_rankings_and_queries():
    index = FakeIndex(["c3", "c1"])
    result = PassageLookup(index, top_k=3).lookup(
        ["maximum database size", "281 TB limit"], run_id="run", allowed_source_ids=None
    )
    assert result.chunks[0].chunk_id == "c3"
    assert len(result.chunks) <= 3


def test_fallback_and_errors():
    vector_only = PassageLookup(FakeIndex(["c3"], documents=False)).lookup(
        ["size"], run_id="run", allowed_source_ids=None
    )
    assert vector_only.fallback == "vector_only"
    assert [c.chunk_id for c in vector_only.chunks] == ["c3"]

    keyword_only = PassageLookup(FakeIndex([], vector_fails=True)).lookup(
        ["281 TB"], run_id="run", allowed_source_ids=None
    )
    assert [c.chunk_id for c in keyword_only.chunks] == ["c3"]
    assert "sk-secret-123" not in keyword_only.error and keyword_only.error

    empty = PassageLookup(FakeIndex([])).lookup(["size"], run_id="nothing", allowed_source_ids=None)
    assert empty.chunks == [] and empty.error == ""


def test_bm25_scores_only_matching_documents():
    bm25 = BM25(["alpha beta", "gamma", "beta beta delta"])
    assert bm25.top("beta", 5) == [2, 0]
    assert bm25.top("missing", 5) == []
