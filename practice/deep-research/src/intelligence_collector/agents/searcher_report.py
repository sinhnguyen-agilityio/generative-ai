"""Searcher gap self-lookup and short report (spec 002, FR-013 to FR-014).

Before a searcher spends web searches, and again before it reports, it looks its evidence points
up in passages already stored for the run's active tasks (``PassageLookup``). Its report is then
drafted by one structured call (role ``searcher_report``) and validated by code: every answered or
partial point must carry a verbatim quote that appears in the cited source's stored text;
anything that cannot be verified becomes ``missing``, and every correction is recorded. The
evidence bundle the writer reads is derived from the report.

The functions here are synchronous building blocks; ``ResearchSupervisor`` decides which run on
worker threads (model calls) and which on its event-loop thread (index reads and VFS writes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from intelligence_collector.agents.prompts import QUERY_REWRITE_TEMPLATE, SEARCHER_REPORT_TEMPLATE
from intelligence_collector.contracts import (
    MAX_QUOTE_WORDS,
    MAX_REPORT_FINDINGS,
    EvidencePoint,
    FailureKind,
    Finding,
    PassageLookupRecord,
    PointReport,
    PointStatus,
    PointSupport,
    QueryRewrite,
    ReportStatus,
    ResearchTodo,
    SearcherReport,
    SearcherReportDraft,
    StopReason,
)
from intelligence_collector.models.redaction import redact_error
from intelligence_collector.observability.tracing import traced
from intelligence_collector.ports import ArtifactStore, ModelRouterPort, VectorIndex
from intelligence_collector.source_verification import quote_in_text, verified_prefix
from intelligence_collector.storage.evidence_index import stored_text_for_source
from intelligence_collector.storage.passage_lookup import PassageLookup

DEFAULT_REPORT_MAX_WORDS = 400
PASSAGE_CHARS = 2400


def task_points(todo: ResearchTodo) -> list[EvidencePoint]:
    """The task's evidence points; a task planned without any reports on its question."""
    return list(todo.evidence_points) or [
        EvidencePoint(point_id="E1", text=todo.research_question[:300])
    ]


@dataclass
class StoredSource:
    """A source the report may cite: where it is, its stored text and the passages shown."""

    source_id: str
    url: str
    title: str
    text: str
    passages: list[str] = field(default_factory=list)
    task_id: str = ""
    artifact_id: str = ""


@dataclass
class LookupFindings:
    """Lookups for a set of evidence points: records to store and the passages they returned."""

    records: list[PassageLookupRecord] = field(default_factory=list)
    passages: dict[str, list[str]] = field(default_factory=dict)  # source_id -> chunk texts
    sources: dict[str, dict[str, str]] = field(default_factory=dict)  # source_id -> metadata
    points: set[str] = field(default_factory=set)  # point ids with at least one passage

    def merge(self, other: "LookupFindings") -> None:
        self.records.extend(other.records)
        for sid, texts in other.passages.items():
            mine = self.passages.setdefault(sid, [])
            mine.extend(t for t in texts if t not in mine)
        for sid, meta in other.sources.items():
            self.sources.setdefault(sid, meta)
        self.points |= other.points


# ----- phrasings and lookups -----


def phrasings_for(
    points: list[EvidencePoint],
    question: str,
    titles: list[str],
    router: ModelRouterPort | None,
    *,
    run_id: str = "",
    thread_id: str = "",
) -> tuple[dict[str, list[str]], bool]:
    """2–3 alternative phrasings per point from one ``query_rewriter`` call (FR-013d). Returns
    the phrasings and whether they are unavailable (lookups then use the combined query only)."""
    if not points or router is None:
        return {}, True
    try:
        rewrite = router.invoke_messages(
            "query_rewriter",
            QUERY_REWRITE_TEMPLATE.format_messages(
                question=question,
                points="\n".join(f"- {p.point_id}: {p.text}" for p in points),
                titles="\n".join(f"- {t}" for t in titles[:10] if t) or "(none yet)",
            ),
            QueryRewrite,
            run_id=run_id,
            thread_id=thread_id,
        )
    except Exception:
        return {}, True
    wanted = {p.point_id for p in points}
    phrasings = {
        item.point_id: [q.strip() for q in item.phrasings if q.strip()][:3]
        for item in rewrite.points
        if item.point_id in wanted
    }
    return phrasings, not phrasings


def lookup_points(
    points: list[EvidencePoint],
    *,
    todo: ResearchTodo,
    question: str,
    lookup: PassageLookup | None,
    run_id: str,
    allowed_source_ids: set[str],
    phrasings: dict[str, list[str]],
    phrasing_failed: bool,
    requested_by: str,
) -> LookupFindings:
    """Look every point up in the allowed stored passages (FR-013b, FR-013d to FR-013f). Only
    reads the index; never waits for running tasks."""
    found = LookupFindings()
    if lookup is None or not points or not allowed_source_ids:
        return found
    for point in points:
        queries = [f"{point.text} — {question}", *phrasings.get(point.point_id, [])]
        result = lookup.lookup(queries, run_id=run_id, allowed_source_ids=allowed_source_ids)
        fallback = result.fallback
        if fallback == "none" and phrasing_failed:
            fallback = "combined_query_only"
        found.records.append(
            PassageLookupRecord(
                requested_by=requested_by,
                point_key=f"{todo.todo_id}:{point.point_id}",
                queries=queries,
                allowed_source_ids=sorted(allowed_source_ids),
                results=result.ranks,
                fallback=fallback,
                error=result.error,
            )
        )
        for chunk in result.chunks:
            texts = found.passages.setdefault(chunk.source_id, [])
            if chunk.text not in texts:
                texts.append(chunk.text)
            found.sources.setdefault(
                chunk.source_id,
                {
                    "url": str(chunk.metadata.get("locator", "")),
                    "task_id": str(chunk.metadata.get("todo_id", "")),
                    "artifact_id": chunk.artifact_id,
                },
            )
            found.points.add(point.point_id)
    return found


def passages_block(found: LookupFindings) -> str:
    """Looked-up passages as text for the searcher's prompt."""
    return "\n\n".join(
        f"Stored source {sid} ({found.sources.get(sid, {}).get('url', '')}):\n"
        + "\n[…]\n".join(texts)
        for sid, texts in found.passages.items()
    )


# ----- sources the report may cite -----


def stored_sources(
    vfs: ArtifactStore,
    index: VectorIndex | None,
    run_id: str,
    evidence_sources: list[dict],
    found: LookupFindings,
) -> dict[str, StoredSource]:
    """The task's accepted sources plus the run's sources the lookups returned, each with the
    stored text quotes are checked against (FR-013a)."""
    chunks_by_source: dict[str, list[str]] = {}
    if index is not None:
        try:
            for chunk in index.documents(run_id):
                chunks_by_source.setdefault(chunk.source_id, []).append(chunk.text)
        except Exception:
            chunks_by_source = {}
    sources: dict[str, StoredSource] = {}
    for entry in evidence_sources:
        sid = str(entry.get("source_id", ""))
        text = "\n".join(chunks_by_source.get(sid, [])) or stored_text_for_source(vfs, run_id, entry)
        excerpt = str(entry.get("excerpt") or "")
        sources[sid] = StoredSource(
            source_id=sid,
            url=str(entry.get("locator", "")),
            title=str((entry.get("provenance") or {}).get("title", "")),
            text="\n".join(t for t in (text, excerpt) if t),
            passages=[excerpt] if excerpt else [],
            task_id=str(entry.get("reused_from_task", "")),
            artifact_id=str(entry.get("artifact_id", "")),
        )
    for sid, texts in found.passages.items():
        source = sources.get(sid)
        if source is None:
            meta = found.sources.get(sid, {})
            source = sources[sid] = StoredSource(
                source_id=sid,
                url=meta.get("url", ""),
                title="",
                text="\n".join(chunks_by_source.get(sid, [])),
                task_id=meta.get("task_id", ""),
                artifact_id=meta.get("artifact_id", ""),
            )
        source.passages.extend(t for t in texts if t not in source.passages)
        source.text = "\n".join(t for t in (source.text, *texts) if t)
    return sources


def _evidence_block(sources: dict[str, StoredSource]) -> str:
    blocks = []
    for sid, source in sources.items():
        passages = "\n[…]\n".join(source.passages)[:PASSAGE_CHARS] or source.text[:PASSAGE_CHARS]
        blocks.append(f"Source {sid}: {source.title or 'Untitled'}\nURL: {source.url}\n{passages}")
    return "\n\n".join(blocks) or "(no sources)"


# ----- report -----


def _words(text: str) -> int:
    return len(text.split())


def empty_report(
    todo: ResearchTodo,
    *,
    stop_reason: str,
    failure_kind: str | None,
    queries: list[str],
    accepted: int,
    rejected: int,
    unreadable: bool = False,
    corrections: list[str] = (),
    lookup_used: set[str] = frozenset(),
) -> SearcherReport:
    """Every point missing: no usable evidence, or an unreadable report."""
    corrections = list(corrections)
    status = (
        ReportStatus.UNREADABLE
        if unreadable
        else ReportStatus.CORRECTED
        if corrections
        else ReportStatus.COMPLETE
    )
    return SearcherReport(
        todo_id=todo.todo_id,
        points=[
            PointReport(
                point_id=p.point_id, status=PointStatus.MISSING, lookup_used=p.point_id in lookup_used
            )
            for p in task_points(todo)
        ],
        accepted_count=accepted,
        rejected_count=rejected,
        queries=list(queries),
        stop_reason=StopReason(stop_reason),
        failure_kind=FailureKind(failure_kind) if failure_kind else None,
        report_status=status,
        corrections=corrections,
    )


def validate_report(
    draft: SearcherReportDraft,
    *,
    todo: ResearchTodo,
    sources: dict[str, StoredSource],
    stop_reason: str,
    failure_kind: str | None,
    queries: list[str],
    accepted: int,
    rejected: int,
    lookup_used: set[str],
    max_words: int = DEFAULT_REPORT_MAX_WORDS,
) -> SearcherReport:
    """Apply contracts/searcher-report.md: known points only, accepted sources only, verified
    quotes, quote and size limits. Never raises for model mistakes; records every correction."""
    corrections: list[str] = []
    wanted = [p.point_id for p in task_points(todo)]
    by_id: dict[str, Any] = {}
    for item in draft.points:
        if item.point_id not in wanted:
            corrections.append(f"{item.point_id}: not an evidence point of this task; dropped")
        elif item.point_id in by_id:
            corrections.append(f"{item.point_id}: reported twice; kept the first")
        else:
            by_id[item.point_id] = item
    points: list[PointReport] = []
    for point_id in wanted:
        item = by_id.get(point_id)
        if item is None:
            corrections.append(f"{point_id}: missing from the report; recorded as missing")
            points.append(
                PointReport(
                    point_id=point_id, status=PointStatus.MISSING, lookup_used=point_id in lookup_used
                )
            )
            continue
        support: list[PointSupport] = []
        for entry in item.support:
            source = sources.get(entry.source_id)
            if source is None:
                corrections.append(
                    f"{point_id}: {entry.source_id} is not an accepted source; support dropped"
                )
                continue
            quote = entry.quote.strip()
            if _words(quote) > MAX_QUOTE_WORDS:
                cut = verified_prefix(quote, source.text, MAX_QUOTE_WORDS)
                if cut:
                    corrections.append(
                        f"{point_id}: quote from {entry.source_id} cut to {MAX_QUOTE_WORDS} words"
                    )
                quote = cut
            if not quote or not quote_in_text(quote, source.text):
                corrections.append(
                    f"{point_id}: quote not found in {entry.source_id}; support dropped"
                )
                continue
            support.append(PointSupport(source_id=entry.source_id, quote=quote))
        status = item.status
        if status != PointStatus.MISSING and not support:
            corrections.append(f"{point_id}: no verified quote; downgraded to missing")
            status = PointStatus.MISSING
        points.append(
            PointReport(
                point_id=point_id,
                status=status,
                support=support if status != PointStatus.MISSING else [],
                lookup_used=point_id in lookup_used,
            )
        )
    findings: list[Finding] = []
    for finding in draft.findings:
        cited = list(dict.fromkeys(finding.source_ids))
        if not cited or any(sid not in sources for sid in cited):
            corrections.append("A finding cited an unknown or rejected source; dropped")
            continue
        findings.append(Finding(text=finding.text.strip(), source_ids=cited))
    if len(findings) > MAX_REPORT_FINDINGS:
        corrections.append(f"Findings cut to {MAX_REPORT_FINDINGS}")
        findings = findings[:MAX_REPORT_FINDINGS]

    def total_words() -> int:
        return sum(_words(f.text) for f in findings) + sum(
            _words(s.quote) for p in points for s in p.support
        )

    while findings and total_words() > max_words:
        findings.pop()
        corrections.append(f"Findings trimmed to stay within {max_words} words")
    return SearcherReport(
        todo_id=todo.todo_id,
        points=points,
        findings=findings,
        accepted_count=accepted,
        rejected_count=rejected,
        queries=list(queries),
        stop_reason=StopReason(stop_reason),
        failure_kind=FailureKind(failure_kind) if failure_kind else None,
        report_status=ReportStatus.CORRECTED if corrections else ReportStatus.COMPLETE,
        corrections=corrections,
    )


def draft_report(
    todo: ResearchTodo,
    question: str,
    sources: dict[str, StoredSource],
    router: ModelRouterPort,
    *,
    run_id: str = "",
    thread_id: str = "",
) -> SearcherReportDraft:
    """One ``searcher_report`` call over the passages read and found. Raises on failure."""
    with traced("searcher.report", metadata={"todo_id": todo.todo_id, "sources": len(sources)}):
        return router.invoke_messages(
            "searcher_report",
            SEARCHER_REPORT_TEMPLATE.format_messages(
                question=question,
                points="\n".join(f"- {p.point_id}: {p.text}" for p in task_points(todo)),
                evidence=_evidence_block(sources),
                max_quote_words=MAX_QUOTE_WORDS,
                max_findings=MAX_REPORT_FINDINGS,
            ),
            SearcherReportDraft,
            run_id=run_id,
            thread_id=thread_id,
        )


def write_report(
    todo: ResearchTodo,
    question: str,
    sources: dict[str, StoredSource],
    router: ModelRouterPort | None,
    *,
    outcome,
    lookup_used: set[str],
    max_words: int = DEFAULT_REPORT_MAX_WORDS,
    run_id: str = "",
    thread_id: str = "",
) -> SearcherReport:
    """Draft and validate the task's report; an unreadable draft reports every point missing."""
    accepted = sum(1 for e in outcome.evaluations if e.get("decision") == "accepted")
    rejected = sum(1 for e in outcome.evaluations if e.get("decision") == "rejected")
    common = {
        "stop_reason": outcome.stop_reason,
        "failure_kind": outcome.failure_kind,
        "queries": outcome.queries,
        "accepted": accepted,
        "rejected": rejected,
        "lookup_used": lookup_used,
    }
    if not sources:
        return empty_report(todo, **common)
    if router is None:
        return empty_report(todo, unreadable=True, corrections=["No report model"], **common)
    try:
        draft = draft_report(todo, question, sources, router, run_id=run_id, thread_id=thread_id)
    except Exception as exc:
        return empty_report(
            todo,
            unreadable=True,
            corrections=[f"Report unreadable: {redact_error(exc)}"],
            **common,
        )
    return validate_report(draft, todo=todo, sources=sources, max_words=max_words, **common)


# ----- evidence bundle derived from the report -----


@dataclass
class BundleFields:
    summary: str
    sufficient: bool
    unresolved_gaps: list[str]
    evidence_sources: list[dict]


def bundle_from_report(
    report: SearcherReport,
    todo: ResearchTodo,
    sources: dict[str, StoredSource],
    evidence_sources: list[dict],
) -> BundleFields:
    """The evidence bundle the writer reads, derived from the report (research.md R7): findings as
    the summary, gaps from missing or partial points, and each cited source's excerpt extended
    with the looked-up passages behind a verified quote. Sources the report cites from other
    tasks' pages are added as references to their capture."""
    texts = {p.point_id: p.text for p in task_points(todo)}
    entries = [dict(entry) for entry in evidence_sources]
    by_sid = {entry["source_id"]: entry for entry in entries}
    for point in report.points:
        for support in point.support:
            source = sources.get(support.source_id)
            if source is None:
                continue
            passages = [t for t in source.passages if quote_in_text(support.quote, t)]
            entry = by_sid.get(support.source_id)
            if entry is None:
                entry = by_sid[support.source_id] = {
                    "source_id": support.source_id,
                    "locator": source.url,
                    "provenance": {"title": source.title},
                    "excerpt": "",
                    "artifact_id": source.artifact_id,
                    "reused_from_task": source.task_id or "lookup",
                }
                entries.append(entry)
            excerpt = str(entry.get("excerpt") or "")
            for text in passages:
                if text not in excerpt:
                    excerpt = f"{excerpt}\n[…]\n{text}" if excerpt else text
            entry["excerpt"] = excerpt
    urls = {sid: s.url for sid, s in sources.items()}
    summary = "\n".join(
        f"- {f.text} " + " ".join(f"[{urls.get(sid, sid)}]" for sid in f.source_ids)
        for f in report.findings
    ) or "\n".join(
        f"- {texts.get(p.point_id, p.point_id)}: "
        + "; ".join(f'"{s.quote}" [{urls.get(s.source_id, s.source_id)}]' for s in p.support)
        for p in report.points
        if p.support
    )
    gaps = [
        texts.get(p.point_id, p.point_id) for p in report.points if p.status != PointStatus.ANSWERED
    ]
    return BundleFields(
        summary=summary or "(no verified findings)",
        sufficient=not gaps,
        unresolved_gaps=gaps,
        evidence_sources=entries,
    )
