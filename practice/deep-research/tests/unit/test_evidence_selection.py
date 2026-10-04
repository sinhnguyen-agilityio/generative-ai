from intelligence_collector.agents.report_writer import select_relevant_evidence
from intelligence_collector.contracts import IndexedChunk
from intelligence_collector.source_verification import catalog_from_bundles
from intelligence_collector.storage.vector_index import LocalVectorIndex
from tests.fixtures.fake_vector import FakeVectorIndex

RUN_ID = "run"


def _bundle(task_id, question, sources):
    return {
        "task_id": task_id,
        "question": question,
        "summary": "summary",
        "sources": [
            {"locator": url, "excerpt": text, "provenance": {"title": url}}
            for url, text in sources
        ],
        "unresolved_gaps": [],
    }


def _index_sources(index, task_id, sources):
    index.index([
        IndexedChunk(
            source_id=url, artifact_id="a", text=text,
            metadata={"run_id": RUN_ID, "todo_id": task_id, "locator": url},
        )
        for url, text in sources
    ])


def test_bundle_at_or_under_top_k_is_left_unfiltered():
    sources = [(f"https://x.test/{n}", "text") for n in range(3)]
    catalog = catalog_from_bundles([_bundle("t1", "Q?", sources)])

    # top_k >= source count: the bundle is returned as-is without ever calling retrieve().
    result = select_relevant_evidence(catalog, FakeVectorIndex(), RUN_ID, top_k=5)

    assert result.bundles[0].source_ids == catalog.bundles[0].source_ids


def test_no_index_returns_the_same_catalog_unfiltered():
    sources = [(f"https://x.test/{n}", "text") for n in range(10)]
    catalog = catalog_from_bundles([_bundle("t1", "Q?", sources)])

    assert select_relevant_evidence(catalog, None, RUN_ID, top_k=3) is catalog


def test_narrows_to_top_k_most_relevant_sources_for_the_question(tmp_path):
    sources = [
        ("https://x.test/locking", "database locking behavior and isolation levels"),
        ("https://x.test/backup", "database backup and restore procedures"),
        ("https://x.test/index", "database index tuning for query performance"),
        ("https://x.test/weather", "the weather forecast for tomorrow is sunny"),
        ("https://x.test/recipe", "a recipe for baking sourdough bread"),
    ]
    catalog = catalog_from_bundles([_bundle("t1", "database locking behavior", sources)])
    index = LocalVectorIndex(tmp_path / "index.json")
    _index_sources(index, "t1", sources)

    result = select_relevant_evidence(catalog, index, RUN_ID, top_k=2, direct_read_chars=0)

    kept_urls = {catalog.sources[sid].url for sid in result.bundles[0].source_ids}
    assert kept_urls == {"https://x.test/locking", "https://x.test/backup"}
    assert len(result.bundles[0].source_ids) == 2
    assert result.sources == catalog.sources  # full source map is untouched, only IDs narrowed


def test_every_bundle_keeps_guaranteed_representation_even_if_less_relevant_globally(tmp_path):
    """Per-bundle scoping: bundle B's sources must never be crowded out by bundle A's."""
    a_sources = [(f"https://a.test/{n}", "highly relevant cache eviction details") for n in range(4)]
    b_sources = [(f"https://b.test/{n}", "mildly related backup topic") for n in range(4)]
    catalog = catalog_from_bundles([
        _bundle("task-a", "cache eviction details", a_sources),
        _bundle("task-b", "backup topic", b_sources),
    ])
    index = LocalVectorIndex(tmp_path / "index.json")
    _index_sources(index, "task-a", a_sources)
    _index_sources(index, "task-b", b_sources)

    result = select_relevant_evidence(catalog, index, RUN_ID, top_k=2, direct_read_chars=0)

    assert len(result.bundles[0].source_ids) == 2
    assert len(result.bundles[1].source_ids) == 2
    assert all(catalog.sources[sid].url.startswith("https://a.test") for sid in result.bundles[0].source_ids)
    assert all(catalog.sources[sid].url.startswith("https://b.test") for sid in result.bundles[1].source_ids)


def test_retrieval_failure_leaves_the_bundle_unfiltered_instead_of_blocking_build():
    class ExplodingIndex:
        def retrieve(self, *a, **k):
            raise RuntimeError("index unavailable")

    sources = [(f"https://x.test/{n}", "text") for n in range(10)]
    catalog = catalog_from_bundles([_bundle("t1", "Q?", sources)])

    result = select_relevant_evidence(catalog, ExplodingIndex(), RUN_ID, top_k=3, direct_read_chars=0)

    assert result.bundles[0].source_ids == catalog.bundles[0].source_ids


def test_unmatched_retrieved_chunks_still_fill_up_to_top_k_from_the_bundle():
    """If retrieval returns chunks that don't map to this bundle's sources, fall back to filling
    the remaining slots from the bundle's own source order rather than returning too few."""
    sources = [(f"https://x.test/{n}", "text") for n in range(5)]
    catalog = catalog_from_bundles([_bundle("t1", "Q?", sources)])
    stray = FakeVectorIndex()  # returns no chunks at all (empty store) -> pure fallback path

    result = select_relevant_evidence(catalog, stray, RUN_ID, top_k=3, direct_read_chars=0)

    assert len(result.bundles[0].source_ids) == 3
    assert set(result.bundles[0].source_ids) <= set(catalog.bundles[0].source_ids)


def test_small_corpus_is_read_directly_without_calling_the_index():
    class NoRetrieval:
        def retrieve(self, *a, **k):
            raise AssertionError("a small corpus must not be retrieved through the index")

    sources = [(f"https://x.test/{n}", "short excerpt") for n in range(10)]
    catalog = catalog_from_bundles([_bundle("t1", "Q?", sources)])

    result = select_relevant_evidence(catalog, NoRetrieval(), RUN_ID, top_k=3)

    assert result.bundles[0].source_ids == catalog.bundles[0].source_ids
