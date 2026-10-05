"""Read the run's stored research for display, without Streamlit: evidence bundles, their
sources and the summary counts shown in the chat."""

from __future__ import annotations

import json
from dataclasses import dataclass

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

    @property
    def summary(self) -> str:
        """The excerpt without the key quotes of an older page summary."""
        return self.excerpt.split("\n\nKey excerpts:", 1)[0].strip()


def todo_bundle(vfs, run_id: str, todo) -> dict:
    """The evidence bundle of a todo's latest research round ({} before it was researched)."""
    todo_id = record_value(todo, "todo_id", "")
    attempt = int(record_value(todo, "research_attempt", 1) or 1)
    prefix = f"research/{todo_id}" + ("" if attempt == 1 else f"/attempts/{attempt}")
    try:
        return json.loads(vfs.read(run_id, f"{prefix}/evidence/result.json"))
    except FileNotFoundError:
        return {}


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
