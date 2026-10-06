"""Report draft artifacts under a run's ``reports/`` folder (the writing itself is in
``agents/report_writer``)."""

from __future__ import annotations

from intelligence_collector.ports import ArtifactStore


def persist_report(vfs: ArtifactStore, run_id: str, report: str, version: int):
    """Save a report draft under the run's ``reports/`` folder.

    Writes ``draft.v{version}.md``, moves the current ``latest.md`` to ``previous.md`` so one
    revision can always be diffed against the last, and rewrites ``latest.md``. Returns the
    ``ArtifactRef`` of ``latest.md``.
    """
    latest_path = "reports/latest.md"
    try:
        previous = vfs.read(run_id, latest_path)
    except FileNotFoundError:
        previous = None
    if previous is not None and previous != report.encode():
        vfs.write(run_id, "reports/previous.md", previous, "previous_draft")
    vfs.write(run_id, f"reports/draft.v{version}.md", report, "draft")
    return vfs.write(run_id, latest_path, report, "latest_report")
