from __future__ import annotations

from intelligence_collector.contracts import IndexedChunk, ResearchGraphState, Stage


def _research_query(scope) -> str:
    return " ".join((
        scope.objective,
        scope.research_boundaries,
        scope.success_criteria,
    ))


def _evidence_chunks(scope, chunks, *, index=None, run_id=None):
    if index is not None:
        chunks = index.retrieve(_research_query(scope), limit=20, run_id=run_id)
    resolved = list(chunks or [])
    if not resolved:
        raise ValueError("synthesis requires indexed evidence with provenance")
    return resolved


def _citation(chunk: IndexedChunk) -> str:
    locator = chunk.metadata.get("locator", "")
    reference = (
        f"evidence:{chunk.chunk_id}; source:{chunk.source_id}; "
        f"artifact:{chunk.artifact_id}"
    )
    if locator:
        reference += f"; locator:{locator}"
    return f"[{reference}]"


def _compose_report(scope, chunks: list[IndexedChunk]) -> str:
    evidence = "\n".join(f"- {chunk.text} {_citation(chunk)}" for chunk in chunks)
    return (
        "# Technical Report\n\n"
        f"## Objective\n{scope.objective}\n\n"
        f"## Research boundaries\n{scope.research_boundaries}\n\n"
        f"## Evidence-backed findings\n{evidence}\n"
    )


def persist_report(vfs, run_id: str, report: str, version: int):
    """Keep previous/latest/versioned drafts; returns the latest-report artifact ref."""
    latest_path = "reports/latest.md"
    try:
        previous = vfs.read(run_id, latest_path)
    except FileNotFoundError:
        previous = None
    if previous is not None:
        vfs.write(run_id, "reports/previous.md", previous, "previous_draft")
    vfs.write(run_id, f"reports/draft.v{version}.md", report, "draft")
    return vfs.write(run_id, latest_path, report, "latest_report")


def synthesize(
    scope,
    chunks=None,
    vfs=None,
    run_id=None,
    version=1,
    *,
    index=None,
):
    evidence = _evidence_chunks(scope, chunks, index=index, run_id=run_id)
    report = _compose_report(scope, evidence)
    if vfs is not None and run_id:
        persist_report(vfs, run_id, report, version)
    return report


def revise_draft(scope, chunks, feedback, vfs=None, run_id=None, version=2):
    evidence = _evidence_chunks(scope, chunks)
    report = _compose_report(scope, evidence)
    report += f"\n## Revision note\n{feedback}\n"
    if vfs is not None and run_id:
        persist_report(vfs, run_id, report, version)
    return report


def revise_draft_node(
    state: ResearchGraphState,
    chunks: list[IndexedChunk],
    feedback: str,
    vfs,
) -> ResearchGraphState:
    """Repair the current draft with existing evidence and require re-verification."""
    if state.scoped_request is None:
        raise ValueError("draft revision requires a scoped request")
    if state.draft:
        vfs.write(state.run_id, "reports/previous.md", state.draft, "previous_draft")
    version = state.recovery_attempt + 2
    revised = revise_draft(
        state.scoped_request,
        chunks,
        feedback,
        vfs,
        state.run_id,
        version,
    )
    return state.model_copy(update={
        "stage": Stage.VERIFYING,
        "status": "active",
        "previous_draft": state.draft,
        "draft": revised,
        "verification": None,
        "approval": None,
        "publication": None,
        "last_error": "",
    })
