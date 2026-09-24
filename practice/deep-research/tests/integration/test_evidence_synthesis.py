import pytest

from intelligence_collector.agents.scope import build_scope
from intelligence_collector.agents.synthesis import synthesize
from intelligence_collector.contracts import IndexedChunk


def _scope():
    return build_scope({
        "objective": "Compare transaction behavior",
        "research_boundaries": "Current local database releases",
        "constraints": "Use indexed primary-source evidence",
        "success_criteria": "Every finding links to stored evidence",
        "deliverable_format": "Technical report",
    })


def _chunk(run_id, text, suffix):
    return IndexedChunk(
        chunk_id=f"chunk-{suffix}",
        source_id=f"source-{suffix}",
        artifact_id=f"artifact-{suffix}",
        text=text,
        metadata={
            "run_id": run_id,
            "locator": f"https://primary.test/{suffix}",
        },
    )


def test_synthesis_retrieves_only_current_run_evidence_and_cites_provenance(stores):
    vfs, index = stores
    relevant = _chunk("run-a", "Transactions are atomic.", "relevant")
    other_run = _chunk("run-b", "Invented cross-run claim.", "other")
    index.index([relevant, other_run])

    report = synthesize(_scope(), vfs=vfs, run_id="run-a", index=index)

    assert "Transactions are atomic." in report
    assert "Invented cross-run claim." not in report
    assert "evidence:chunk-relevant" in report
    assert "source:source-relevant" in report
    assert "artifact:artifact-relevant" in report
    assert "locator:https://primary.test/relevant" in report


def test_synthesis_preserves_previous_and_latest_report_versions(stores):
    vfs, index = stores
    first = _chunk("versioned", "First evidence version.", "v1")
    index.index([first])
    report_v1 = synthesize(
        _scope(), vfs=vfs, run_id="versioned", version=1, index=index
    )
    second = _chunk("versioned", "Second evidence version.", "v2")
    index.index([second])

    report_v2 = synthesize(
        _scope(), vfs=vfs, run_id="versioned", version=2, index=index
    )

    assert vfs.read("versioned", "reports/previous.md").decode() == report_v1
    assert vfs.read("versioned", "reports/latest.md").decode() == report_v2
    assert vfs.read("versioned", "reports/draft.v1.md").decode() == report_v1
    assert vfs.read("versioned", "reports/draft.v2.md").decode() == report_v2


def test_synthesis_refuses_to_generate_factual_report_without_evidence(stores):
    vfs, index = stores

    with pytest.raises(ValueError, match="indexed evidence"):
        synthesize(_scope(), vfs=vfs, run_id="empty", index=index)

    assert not (vfs.root / "empty" / "reports").exists()
