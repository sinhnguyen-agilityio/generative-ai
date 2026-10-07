"""Research task execution: run one research sub-agent and persist what it gathered.

The sub-agent (``researcher.py``) only searches and compresses. This module turns its
``ResearchOutcome`` into durable, run-scoped VFS artifacts and vector-index records. Each page's
full text is stored once as ``raw/<n>.md`` next to its capture record ``raw/<n>.json``; the index
receives that page's chunks, never the whole page as one record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from intelligence_collector.ports import ArtifactStore, VectorIndex
from intelligence_collector.contracts import ArtifactRef

from intelligence_collector.agents.researcher import ResearchAgent, ResearchOutcome
from intelligence_collector.contracts import (
    IndexedChunk,
    ResearchTodo,
    SourceRecord,
    TodoStatus,
)
from intelligence_collector.storage.evidence_index import mark_indexed, source_chunks
from intelligence_collector.tools.page_content import source_id_for
from intelligence_collector.tools.url_tracking import UrlRegistry, normalize_url

# Sub-agent fields that are page content or bookkeeping, not provenance.
_NOT_PROVENANCE = {"raw_content"}


def _evidence_path(todo: ResearchTodo) -> str:
    return f"{research_prefix(todo)}/evidence/result.json"


def research_prefix(todo: ResearchTodo) -> str:
    base = f"research/{todo.todo_id}"
    return base if todo.research_attempt == 1 else f"{base}/attempts/{todo.research_attempt}"


def already_researched(vfs: ArtifactStore, run_id: str, todo: ResearchTodo) -> bool:
    """Idempotent resume guard: an evidence bundle means the task finished earlier."""
    try:
        vfs.read(run_id, _evidence_path(todo))
    except FileNotFoundError:
        return False
    return True


def restore_research(vfs: ArtifactStore, run_id: str, todo: ResearchTodo):
    """Recover the evidence reference if a crash preceded the graph checkpoint write."""
    path = _evidence_path(todo)
    text = vfs.read(run_id, path)
    bundle = json.loads(text)
    if bundle.get("task_id") != todo.todo_id:
        raise ValueError("Saved evidence belongs to a different research task")
    todo.status = TodoStatus.COMPLETED
    ref = ArtifactRef(
        run_id=run_id,
        path=str(vfs.root / run_id / path),
        kind="evidence_bundle",
        content_hash=hashlib.sha256(text).hexdigest(),
        task_id=todo.todo_id,
    )
    # Bundles list chunk records without their text (the pages hold it).
    chunks = [
        IndexedChunk.model_validate({"text": "", **item})
        for item in bundle.get("indexed_chunks", [])
    ]
    return todo, [ref], chunks


def _write_failure(vfs: ArtifactStore, run_id: str, todo: ResearchTodo, message: str, *, technical=True):
    return vfs.write_json(
        run_id,
        f"{research_prefix(todo)}/outcome.json",
        {
            "status": "failed",
            "question": todo.research_question,
            "error": message,
            "failure_kind": "technical" if technical else "no_evidence",
        },
        "research_outcome",
        task_id=todo.todo_id,
    )


def failed_outcome(todo: ResearchTodo, error: str) -> ResearchOutcome:
    """Outcome for a sub-agent that crashed before returning (e.g. model unavailable)."""
    return ResearchOutcome(
        question=todo.research_question,
        search_budget=0,
        mcp_server="",
        mcp_tool="",
        errors=[error],
    )


@dataclass
class PersistedPages:
    """What ``persist_pages`` stored for one task: artifact refs, indexed chunks, and one evidence
    entry per accepted source (in source order)."""

    refs: list[ArtifactRef] = field(default_factory=list)
    chunks: list[IndexedChunk] = field(default_factory=list)
    evidence_sources: list[dict] = field(default_factory=list)
    reused: int = 0


def _write_evaluations(vfs: ArtifactStore, run_id: str, todo: ResearchTodo, outcome) -> ArtifactRef:
    """Every result decision of the task, its queries and why it stopped (FR-012)."""
    return vfs.write_json(
        run_id,
        f"{research_prefix(todo)}/evaluations.json",
        {
            "task_id": todo.todo_id,
            "research_attempt": todo.research_attempt,
            "queries": list(outcome.queries),
            "stop_reason": outcome.stop_reason,
            "failure_kind": outcome.failure_kind,
            "evaluations": list(outcome.evaluations),
            "corrections": list(outcome.corrections),
            "errors": list(outcome.errors),
        },
        "result_evaluations",
        task_id=todo.todo_id,
    )


def persist_pages(
    vfs: ArtifactStore,
    index: VectorIndex | None,
    run_id: str,
    todo: ResearchTodo,
    outcome: ResearchOutcome,
    registry: UrlRegistry | None = None,
) -> PersistedPages:
    """Store the task's result evaluations, then each accepted page (raw capture and cleaned
    text) and index every chunk of it, before the task reports (FR-009, FR-012).

    Each URL is captured and indexed once per run: a URL already in the run's ``UrlRegistry``
    (from another task, a re-scoped plan, or an earlier invocation) is referenced by its first
    capture instead of being written and indexed again. Pass a shared ``registry`` when
    persisting several tasks; otherwise it is loaded from and saved to the VFS here.
    """
    stored = PersistedPages(refs=[_write_evaluations(vfs, run_id, todo, outcome)])
    sources = outcome.unique_sources
    if not sources:
        return stored
    snippet_only = {
        normalize_url(str(item.get("url", ""))): bool(item.get("snippet_only"))
        for item in outcome.evaluations
        if item.get("decision") == "accepted"
    }
    urls = registry if registry is not None else UrlRegistry.load(vfs, run_id)
    for i, source in enumerate(sources):
        # Underscore keys (_query, _covered_by) are searcher bookkeeping, not page provenance.
        payload = {
            key: value
            for key, value in source.items()
            if not key.startswith("_") and key not in _NOT_PROVENANCE
        }
        record = SourceRecord(
            source_id=source_id_for(source["url"]),
            locator=source["url"],
            mcp_server=outcome.mcp_server,
            mcp_tool=outcome.mcp_tool,
            provenance={"query": source.get("_query", ""), **payload},
        )
        # What the LLMs read: the page's selected passages, or the search snippet without a
        # usable page.
        excerpt = source.get("content", "")
        page = source.get("raw_content") or ""
        entry = {
            "source_id": record.source_id,
            "locator": record.locator,
            "retrieved_at": record.retrieved_at.isoformat(),
            "mcp_server": record.mcp_server,
            "mcp_tool": record.mcp_tool,
            "provenance": record.provenance,
            "excerpt": excerpt,
            "snippet_only": snippet_only.get(normalize_url(record.locator), False),
        }
        known = urls.get(record.locator)
        if known is not None:
            # Already captured in this run: point at the first capture, skip write + index.
            stored.reused += 1
            stored.evidence_sources.append(
                {
                    **entry,
                    "source_id": known["source_id"],
                    "artifact_id": known["artifact_id"],
                    "content_hash": known["content_hash"],
                    "reused_from_task": known["task_id"],
                }
            )
            continue
        raw_path = None
        if page:
            raw_path = f"{research_prefix(todo)}/raw/{i}.md"
            stored.refs.append(vfs.write(run_id, raw_path, page, "raw_page", task_id=todo.todo_id))
        ref = vfs.write_json(
            run_id,
            f"{research_prefix(todo)}/raw/{i}.json",
            {
                "source": record.model_dump(mode="json"),
                "content": excerpt,
                "raw_path": raw_path,
                "snippet_only": entry["snippet_only"],
            },
            "raw_capture",
            task_id=todo.todo_id,
        )
        stored.refs.append(ref)
        stored.chunks.extend(
            source_chunks(
                run_id=run_id,
                todo_id=todo.todo_id,
                source_id=record.source_id,
                artifact_id=ref.artifact_id,
                locator=record.locator,
                title=str(source.get("title", "")),
                text=page or excerpt,
            )
        )
        stored.evidence_sources.append(
            {
                **entry,
                "artifact_id": ref.artifact_id,
                "content_hash": ref.content_hash,
                "raw_path": raw_path,
            }
        )
        urls.register(
            record.locator,
            {
                "source_id": record.source_id,
                "artifact_id": ref.artifact_id,
                "content_hash": ref.content_hash,
                "task_id": todo.todo_id,
                "locator": record.locator,
            },
        )
    if stored.chunks and index is not None:
        index.index(stored.chunks)
        mark_indexed(vfs, run_id, index)
    registry_ref = urls.save(vfs, run_id)
    if registry_ref is not None:
        stored.refs.append(registry_ref)
    return stored


def write_evidence_bundle(
    vfs: ArtifactStore,
    run_id: str,
    todo: ResearchTodo,
    outcome: ResearchOutcome,
    stored: PersistedPages,
    *,
    summary: str,
    sufficient: bool,
    unresolved_gaps: list[str],
    extra: dict | None = None,
):
    """Write the task's notes, excerpts and evidence bundle — the last step of a task, so an
    existing bundle means the task is done (``already_researched``). Returns ``(todo, refs,
    chunks)`` with the todo COMPLETED."""
    todo.status = TodoStatus.COMPLETED
    refs = list(stored.refs)
    sources = stored.evidence_sources
    refs.append(
        vfs.write(
            run_id,
            f"{research_prefix(todo)}/notes/result.md",
            f"# Research notes\n\nQuestion: {todo.research_question}\n\n"
            f"Search calls: {outcome.search_calls}/{outcome.search_budget}\n\n"
            f"Sources: {len(sources)} ({stored.reused} already captured earlier in this run)\n\n"
            f"## Findings\n\n{summary}\n",
            "notes",
            task_id=todo.todo_id,
        )
    )
    refs.append(
        vfs.write_json(
            run_id,
            f"{research_prefix(todo)}/excerpts/result.json",
            {"excerpts": [item["excerpt"] for item in sources]},
            "excerpts",
            task_id=todo.todo_id,
        )
    )
    refs.append(
        vfs.write_json(
            run_id,
            _evidence_path(todo),
            {
                "task_id": todo.todo_id,
                "research_attempt": todo.research_attempt,
                "question": todo.research_question,
                "summary": summary,
                "sources": sources,
                "search_calls": outcome.search_calls,
                "search_budget": outcome.search_budget,
                "sufficient": sufficient,
                "unresolved_gaps": list(unresolved_gaps),
                "indexed_chunks": [
                    chunk.model_dump(mode="json", exclude={"text"}) for chunk in stored.chunks
                ],
                **(extra or {}),
            },
            "evidence_bundle",
            task_id=todo.todo_id,
        )
    )
    return todo, refs, stored.chunks


def write_task_failure(
    vfs: ArtifactStore, run_id: str, todo: ResearchTodo, outcome: ResearchOutcome, stored
):
    """Record a task that produced no usable evidence; returns ``(todo, refs, [])``."""
    todo.status = TodoStatus.FAILED
    reason = "; ".join(outcome.errors) or "Research returned no sources"
    failure = _write_failure(vfs, run_id, todo, reason, technical=bool(outcome.errors))
    return todo, [*(stored.refs if stored else []), failure], []


def persist_research(
    vfs: ArtifactStore,
    index: VectorIndex | None,
    run_id: str,
    todo: ResearchTodo,
    outcome: ResearchOutcome,
    registry: UrlRegistry | None = None,
):
    """``persist_pages`` then the evidence bundle from ``outcome.findings``, in one step.

    A task is COMPLETED when it gathered sources and summarized findings, even if the findings
    report unresolved gaps (recorded in the evidence bundle; the writer discloses them). It is
    FAILED only when retrieval or summarizing produced nothing usable.
    """
    stored = persist_pages(vfs, index, run_id, todo, outcome, registry)
    if not stored.evidence_sources or outcome.findings is None:
        return write_task_failure(vfs, run_id, todo, outcome, stored)
    findings = outcome.findings
    return write_evidence_bundle(
        vfs,
        run_id,
        todo,
        outcome,
        stored,
        summary=findings.summary,
        sufficient=findings.sufficient,
        unresolved_gaps=findings.unresolved_gaps,
    )


def run_worker(
    vfs: ArtifactStore,
    index: VectorIndex | None,
    run_id: str,
    todo: ResearchTodo, researcher: ResearchAgent, *, thread_id=""):
    """Research one todo with its own searcher, on the calling thread: lookup, search, persist,
    report and evidence bundle (``search_task.SearchTask``)."""
    from intelligence_collector.agents.search_task import SearchTask  # imports this module
    from intelligence_collector.storage.passage_lookup import PassageLookup

    if already_researched(vfs, run_id, todo):
        return restore_research(vfs, run_id, todo)
    task = SearchTask(
        vfs,
        index,
        researcher,
        run_id,
        todo,
        todo.research_question,
        {todo.todo_id},
        thread_id=thread_id,
        lookup=PassageLookup(index) if index is not None else None,
    )
    return task.run()
