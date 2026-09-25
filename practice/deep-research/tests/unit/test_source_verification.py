from intelligence_collector.contracts import IssueType
from intelligence_collector.source_verification import (
    catalog_from_bundles,
    failure_fingerprint,
    gap_id,
    verify_report,
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


def _failed(result):
    return {check.name for check in result.check_results if not check.passed}


GOOD_CLAIM = "A cache stores reusable data for later use [S1]."


def test_well_cited_grounded_report_passes_every_check():
    report = cited_report(
        [GOOD_CLAIM, "Version 3.13 added 2 storage tiers [S1]."], {"S1": URL}
    )

    result = verify_report(report, catalog_from_bundles([_bundle()]))

    assert result.passed, result.repair_details
    assert result.issue_type is None
    assert failure_fingerprint(result) == ""
    assert "claims_grounded" in result.checks and len(result.checks) == 10


def test_uncited_claim_is_a_draft_defect():
    report = cited_report([GOOD_CLAIM, "Caches also make every workload faster overall."], {"S1": URL})

    result = verify_report(report, catalog_from_bundles([_bundle()]))

    assert _failed(result) == {"claims_cited"}
    assert result.issue_type == IssueType.DRAFT_DEFECT
    assert "Uncited claim" in result.repair_details


def test_unknown_citation_and_invented_url_are_rejected():
    report = cited_report(
        [GOOD_CLAIM, "More detail is at https://invented.test/page for caching [S9]."],
        {"S1": URL},
    )

    result = verify_report(report, catalog_from_bundles([_bundle()]))

    assert {"citation_ids_known", "urls_known"} <= _failed(result)


def test_numbers_must_appear_in_the_cited_source():
    report = cited_report([GOOD_CLAIM, "Version 3.14 added 2 storage tiers [S1]."], {"S1": URL})

    result = verify_report(report, catalog_from_bundles([_bundle()]))

    assert _failed(result) == {"claims_grounded"}
    assert "3.14" in result.repair_details


def test_claim_wording_must_overlap_the_cited_source():
    report = cited_report(
        [GOOD_CLAIM, "Quantum annealing accelerates cryptographic factorization benchmarks [S1]."],
        {"S1": URL},
    )

    result = verify_report(report, catalog_from_bundles([_bundle()]))

    assert _failed(result) == {"claims_grounded"}
    assert "not supported by [S1]" in result.repair_details


def test_sources_section_must_list_cited_ids_with_exact_urls():
    report = cited_report([GOOD_CLAIM], {"S1": "https://docs.test/other"})
    catalog = catalog_from_bundles([_bundle(), _bundle("t2", "Other?", "https://docs.test/other")])

    result = verify_report(report, catalog)

    assert "sources_section" in _failed(result)
    assert "its URL is https://docs.test/cache" in result.repair_details


def test_structure_requires_title_body_and_sources():
    result = verify_report(GOOD_CLAIM, catalog_from_bundles([_bundle()]))

    assert "structure" in _failed(result)


def test_every_researched_question_must_be_cited():
    catalog = catalog_from_bundles([
        _bundle(),
        _bundle("t2", "How are caches invalidated?", "https://docs.test/invalidate"),
    ])

    result = verify_report(cited_report([GOOD_CLAIM], {"S1": URL}), catalog)

    assert _failed(result) == {"question_coverage"}
    assert "question:t2" in result.affected_ids


def test_unresearched_gap_is_the_primary_evidence_gap():
    catalog = catalog_from_bundles([_bundle(gaps=["cache eviction policy details"])])
    report = cited_report(["Caches are always faster everywhere."], {"S1": URL})

    result = verify_report(report, catalog)

    assert result.issue_type == IssueType.EVIDENCE_GAP  # outranks the draft defects
    assert gap_id("t1", "cache eviction policy details") in result.affected_ids


def test_researched_gap_must_then_be_disclosed_under_limitations():
    catalog = catalog_from_bundles([_bundle(gaps=["cache eviction policy details"])])
    researched = [gap_id("t1", "cache eviction policy details")]

    undisclosed = verify_report(cited_report([GOOD_CLAIM], {"S1": URL}), catalog, researched_gaps=researched)
    disclosed = verify_report(
        cited_report([GOOD_CLAIM], {"S1": URL}, limitations=["Cache eviction policy details remain unknown."]),
        catalog,
        researched_gaps=researched,
    )

    assert _failed(undisclosed) == {"gaps_disclosed"}
    assert disclosed.passed, disclosed.repair_details


def test_missing_evidence_is_an_evidence_gap():
    result = verify_report(cited_report([GOOD_CLAIM], {"S1": URL}), catalog_from_bundles([]))

    assert result.issue_type == IssueType.EVIDENCE_GAP
    assert "evidence_present" in _failed(result)


def test_fingerprint_tracks_what_failed_not_the_draft_hash():
    catalog = catalog_from_bundles([_bundle()])
    uncited = "Caches also make every workload faster overall."
    first = verify_report(cited_report([GOOD_CLAIM, uncited], {"S1": URL}), catalog)
    reworded_elsewhere = verify_report(
        cited_report(["A cache stores reusable data [S1].", uncited], {"S1": URL}), catalog
    )
    different = verify_report(cited_report([GOOD_CLAIM, "Another unsupported sentence here."], {"S1": URL}), catalog)

    assert first.artifact_hash != reworded_elsewhere.artifact_hash
    assert failure_fingerprint(first) == failure_fingerprint(reworded_elsewhere)
    assert failure_fingerprint(first) != failure_fingerprint(different)
