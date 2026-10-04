from intelligence_collector.contracts import VerificationCheck, VerificationResult
from intelligence_collector.source_verification import (
    catalog_from_bundles,
    cited_source_excerpts,
    source_context,
    failure_fingerprint,
)
from tests.fixtures.fake_models import cited_report

URL = "https://docs.test/cache"


def _bundle(task_id="t1", question="How do caches store data?", url=URL, gaps=()):
    return {
        "task_id": task_id,
        "question": question,
        "summary": f"Caches store reusable data [{url}]",
        "sources": [{
            "locator": url,
            "excerpt": "A cache stores reusable data. Version 3.13 added 2 storage tiers.",
            "provenance": {"title": "Cache docs"},
        }],
        "unresolved_gaps": list(gaps),
    }


GOOD_CLAIM = "A cache stores reusable data for later use [S1]."


def _failed_result(artifact_hash, *failing):
    """A failed verification whose failing checks are (name, affected_ids) pairs."""
    return VerificationResult(
        artifact_hash=artifact_hash,
        passed=False,
        check_results=[
            VerificationCheck(name=name, passed=False, affected_ids=list(ids))
            for name, ids in failing
        ],
    )


def test_fingerprint_tracks_what_failed_not_the_draft_hash():
    first = _failed_result("hash-1", ("criterion_1", ["criterion:abc"]))
    reworded = _failed_result("hash-2", ("criterion_1", ["criterion:abc"]))
    different = _failed_result("hash-1", ("criterion_2", ["criterion:def"]))

    assert failure_fingerprint(first) == failure_fingerprint(reworded)
    assert failure_fingerprint(first) != failure_fingerprint(different)


def test_catalog_assigns_stable_ids_per_unique_url_and_keeps_gaps():
    catalog = catalog_from_bundles([
        _bundle(gaps=["No benchmark data"]),
        _bundle(task_id="t2", question="How is data evicted?"),  # same URL again
    ])

    assert list(catalog.sources) == ["S1"]
    assert [bundle.source_ids for bundle in catalog.bundles] == [["S1"], ["S1"]]
    assert [(gap.todo_id, gap.text) for gap in catalog.gaps] == [("t1", "No benchmark data")]


def test_cited_source_excerpts_returns_only_the_excerpts_the_report_actually_cites():
    other_bundle = _bundle(task_id="t2", question="How is data evicted?", url="https://docs.test/evict")
    catalog = catalog_from_bundles([_bundle(), other_bundle])

    report = cited_report([GOOD_CLAIM], {"S1": URL})  # cites only S1, never S2

    assert cited_source_excerpts(report, catalog) == [source_context(catalog.sources["S1"])]


def test_cited_source_excerpts_deduplicates_repeated_citations():
    catalog = catalog_from_bundles([_bundle()])
    report = cited_report([GOOD_CLAIM, "Version 3.13 added 2 storage tiers [S1]."], {"S1": URL})

    assert cited_source_excerpts(report, catalog) == [source_context(catalog.sources["S1"])]


def test_cited_source_excerpts_is_empty_when_nothing_is_cited():
    catalog = catalog_from_bundles([_bundle()])

    assert cited_source_excerpts("# Report\n\nNo citations here.", catalog) == []
