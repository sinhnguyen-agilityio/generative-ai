"""Read the run's stored research for display, without Streamlit: evidence bundles, their
sources, each task's searches and report, the research rounds, the claim verdicts of a draft and
the summary counts shown in the chat."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

from intelligence_collector.agents.supervisor import read_supervisor_log
from intelligence_collector.ui.run_status import record_value


@dataclass(frozen=True)
class SourceView:
    """One collected page as the UI shows it."""

    locator: str
    title: str
    excerpt: str  # what the LLMs read: the page's selected passages, or the search snippet
    page_stored: bool  # the full page is in the VFS and indexed
    linked: bool  # captured by another task and only referenced here
    snippet_only: bool = False  # accepted on its search snippet: no usable page (FR-012)

    @property
    def summary(self) -> str:
        """The excerpt without the key quotes of an older page summary."""
        return self.excerpt.split("\n\nKey excerpts:", 1)[0].strip()


def _prefix(todo) -> str:
    todo_id = record_value(todo, "todo_id", "")
    attempt = int(record_value(todo, "research_attempt", 1) or 1)
    return f"research/{todo_id}" + ("" if attempt == 1 else f"/attempts/{attempt}")


def _read_json(vfs, run_id: str, path: str):
    try:
        return json.loads(vfs.read(run_id, path))
    except (FileNotFoundError, ValueError):
        return None


def todo_bundle(vfs, run_id: str, todo) -> dict:
    """The evidence bundle of a todo's latest research attempt ({} before it was researched)."""
    return _read_json(vfs, run_id, f"{_prefix(todo)}/evidence/result.json") or {}


def bundle_sources(bundle: dict) -> list[SourceView]:
    """The distinct sources of one evidence bundle, in capture order."""
    seen: set[str] = set()
    sources = []
    for source in bundle.get("sources", []):
        locator = source.get("locator")
        if not locator or locator in seen:
            continue
        seen.add(locator)
        sources.append(
            SourceView(
                locator=locator,
                title=(source.get("provenance") or {}).get("title") or locator,
                excerpt=str(source.get("excerpt") or ""),
                page_stored=bool(source.get("raw_path")),
                linked=bool(source.get("reused_from_task")),
                snippet_only=bool(source.get("snippet_only")),
            )
        )
    return sources


@dataclass(frozen=True)
class ResearchCounts:
    tasks: int
    sources: int
    follow_ups: int


def research_counts(vfs, run_id: str, todos: list) -> ResearchCounts:
    """Tasks, distinct sources and supervisor follow-ups of a finished research phase."""
    sources = {
        source.locator
        for todo in todos
        for source in bundle_sources(todo_bundle(vfs, run_id, todo))
    }
    follow_ups = sum(
        bool(task.get("follow_up"))
        for entry in read_supervisor_log(vfs, run_id)
        if entry.get("kind") == "delegate"
        for task in entry.get("tasks") or []
    )
    return ResearchCounts(tasks=len(todos), sources=len(sources), follow_ups=follow_ups)


@dataclass(frozen=True)
class TaskResearch:
    """How one task searched and what it reported (``evaluations.json`` and ``report.json``)."""

    queries: list[str]
    accepted: int
    rejected: dict[str, int]  # reject reason -> results
    stop_reason: str
    report_status: str
    open_points: list[str] = field(default_factory=list)  # "E2 partial"
    corrections: list[str] = field(default_factory=list)


def task_research(vfs, run_id: str, todo) -> TaskResearch | None:
    """The searches, result decisions, stop reason and report of a task; None before it ran."""
    prefix = _prefix(todo)
    evaluations = _read_json(vfs, run_id, f"{prefix}/evaluations.json")
    report = _read_json(vfs, run_id, f"{prefix}/report.json")
    if evaluations is None and report is None:
        return None
    evaluations = evaluations or {}
    report = report or {}
    decisions = list(evaluations.get("evaluations") or [])
    return TaskResearch(
        queries=list(report.get("queries") or evaluations.get("queries") or []),
        accepted=sum(item.get("decision") == "accepted" for item in decisions),
        rejected=dict(
            Counter(
                str(item.get("reason") or "unspecified")
                for item in decisions
                if item.get("decision") == "rejected"
            )
        ),
        stop_reason=str(report.get("stop_reason") or evaluations.get("stop_reason") or ""),
        report_status=str(report.get("report_status") or ""),
        open_points=[
            f"{point.get('point_id')} {point.get('status')}"
            for point in report.get("points") or []
            if point.get("status") != "answered"
        ],
        corrections=list(report.get("corrections") or []),
    )


@dataclass(frozen=True)
class FollowUpView:
    question: str
    gap_point_ids: list[str]


@dataclass(frozen=True)
class RoundView:
    """One research round's gap analysis (``research/rounds/round-{n}.json``)."""

    round: int
    ended: str
    closed_essential: int
    open_essential: list[str]  # "t2:E1 — text"
    follow_ups: list[FollowUpView]
    refusals: list[str]


def research_rounds(vfs, run_id: str) -> list[RoundView]:
    """Every analyzed research round, in order."""
    rounds = []
    n = 1
    while (saved := _read_json(vfs, run_id, f"research/rounds/round-{n}.json")) is not None:
        analysis = saved.get("analysis") or {}
        rounds.append(
            RoundView(
                round=n,
                ended=str(analysis.get("ended") or ""),
                closed_essential=int(analysis.get("closed_essential") or 0),
                open_essential=[
                    f"{gap.get('point_key')} — {gap.get('text', '')}"
                    for gap in analysis.get("gaps") or []
                    if gap.get("essential") and gap.get("resolution") == "open"
                ],
                follow_ups=[
                    FollowUpView(
                        question=str(todo.get("research_question", "")),
                        gap_point_ids=list(todo.get("gap_point_ids") or []),
                    )
                    for todo in saved.get("follow_up_todos") or []
                ],
                refusals=[str(r.get("reason", "")) for r in analysis.get("refusals") or []],
            )
        )
        n += 1
    return rounds


@dataclass(frozen=True)
class CitedText:
    """The exact text of one cited source the claim judge read for this draft."""

    source_id: str
    title: str
    url: str
    text: str
    snippet_only: bool


@dataclass(frozen=True)
class ClaimView:
    claim_id: str
    sentence: str
    verdict: str
    failures: list[str]
    sources: list[CitedText]

    @property
    def passed(self) -> bool:
        return not self.failures


def snippet_only_urls(vfs, run_id: str, todos: list) -> set[str]:
    """Locators accepted on their search snippet only, across the run's evidence bundles."""
    return {
        source.locator
        for todo in todos
        for source in bundle_sources(todo_bundle(vfs, run_id, todo))
        if source.snippet_only
    }


def claim_views(vfs, run_id: str, snapshot: dict) -> list[ClaimView]:
    """The judged factual sentences of the verified draft, failing ones first, each with the
    snapshot text (``synthesis/evidence.v{N}.json``) of every source it cites (SC-006)."""
    ref = snapshot.get("claim_evaluation_ref")
    evaluation = _read_json(vfs, run_id, ref) if ref and run_id else None
    if not evaluation:
        return []
    version = evaluation.get("draft_version")
    texts = _read_json(vfs, run_id, f"synthesis/evidence.v{version}.json") or {}
    snippets = snippet_only_urls(vfs, run_id, list(snapshot.get("todos") or []))
    views = []
    for claim in evaluation.get("claims") or []:
        if not claim.get("states_fact", True):
            continue
        sources = []
        for sid in claim.get("cited_source_ids") or []:
            stored = texts.get(sid) or {}
            url = str(stored.get("url", ""))
            sources.append(
                CitedText(
                    source_id=sid,
                    title=str(stored.get("title") or url or sid),
                    url=url,
                    text=str(stored.get("text", "")),
                    snippet_only=url in snippets,
                )
            )
        views.append(
            ClaimView(
                claim_id=str(claim.get("claim_id", "")),
                sentence=str(claim.get("sentence", "")),
                verdict=str(claim.get("verdict") or ""),
                failures=list(claim.get("failures") or []),
                sources=sources,
            )
        )
    return sorted(views, key=lambda view: view.passed)
