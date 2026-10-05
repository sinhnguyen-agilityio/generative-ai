"""Research task execution: run one research sub-agent and persist what it gathered.

The sub-agent (``researcher.py``) only searches and compresses. This module turns its
``ResearchOutcome`` into durable, run-scoped VFS artifacts and vector-index records. Each page's
full text is stored once as ``raw/<n>.md`` next to its capture record ``raw/<n>.json``; the index
receives that page's chunks, never the whole page as one record.
"""

from __future__ import annotations

import hashlib
import json

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
from intelligence_collector.tools.url_tracking import UrlRegistry

# Sub-agent fields that are page content or bookkeeping, not provenance.
_NOT_PROVENANCE = {"raw_content"}


def _evidence_path(todo: ResearchTodo) -> str:
    return f"{research_prefix(todo)}/evidence/result.json"


def research_prefix(todo: ResearchTodo) -> str:
    base = f"research/{todo.todo_id}"
    return base if todo.research_attempt == 1 else f"{base}/attempts/{todo.research_attempt}"


def already_researched(vfs, run_id: str, todo: ResearchTodo) -> bool:
    """Idempotent resume guard: an evidence bundle means the task finished earlier."""
    try:
        vfs.read(run_id, _evidence_path(todo))
    except FileNotFoundError:
        return False
    return True


def restore_research(vfs, run_id: str, todo: ResearchTodo):
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


def _write_failure(vfs, run_id: str, todo: ResearchTodo, message: str, *, technical=True):
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


def persist_research(
    vfs,
    index,
    run_id: str,
    todo: ResearchTodo,
    outcome: ResearchOutcome,
    registry: UrlRegistry | None = None,
):
    """Write raw captures and pages, notes, excerpts, and the evidence bundle; index each page's
    chunks.

    Each URL is captured and indexed once per run: a URL already in the run's ``UrlRegistry``
    (from another task, a re-scoped plan, or an earlier invocation) is referenced by its first
    capture instead of being written and indexed again. Pass a shared ``registry`` when
    persisting several tasks; otherwise it is loaded from and saved to the VFS here.

    A task is COMPLETED when it gathered sources and compressed findings, even if the findings
    report unresolved gaps (recorded in the evidence bundle; the writer discloses them). It is
    FAILED only when retrieval or compression produced nothing usable.
    """
    sources = outcome.unique_sources
    if not sources or outcome.findings is None:
        todo.status = TodoStatus.FAILED
        reason = "; ".join(outcome.errors) or "Research returned no sources"
        return todo, [_write_failure(vfs, run_id, todo, reason, technical=bool(outcome.errors))], []

    urls = registry if registry is not None else UrlRegistry.load(vfs, run_id)
    refs, chunks, evidence_sources = [], [], []
    reused = 0
    for i, source in enumerate(sources):
        # Underscore keys (_query, _covered_by) are sub-agent bookkeeping, not page provenance.
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
        # What the LLMs read: the page summary, or the search snippet without a usable page.
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
        }
        known = urls.get(record.locator)
        if known is not None:
            # Already captured in this run: point at the first capture, skip write + index.
            reused += 1
            evidence_sources.append(
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
            refs.append(vfs.write(run_id, raw_path, page, "raw_page", task_id=todo.todo_id))
        ref = vfs.write_json(
            run_id,
            f"{research_prefix(todo)}/raw/{i}.json",
            {"source": record.model_dump(mode="json"), "content": excerpt, "raw_path": raw_path},
            "raw_capture",
            task_id=todo.todo_id,
        )
        refs.append(ref)
        chunks.extend(
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
        evidence_sources.append(
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
    if chunks:
        index.index(chunks)
        mark_indexed(vfs, run_id, index)
    registry_ref = urls.save(vfs, run_id)
    if registry_ref is not None:
        refs.append(registry_ref)

    findings = outcome.findings
    todo.status = TodoStatus.COMPLETED
    refs.append(
        vfs.write(
            run_id,
            f"{research_prefix(todo)}/notes/result.md",
            f"# Research notes\n\nQuestion: {todo.research_question}\n\n"
            f"Search calls: {outcome.search_calls}/{outcome.search_budget}\n\n"
            f"Sources: {len(sources)} ({reused} already captured earlier in this run)\n\n"
            f"## Findings\n\n{findings.summary}\n",
            "notes",
            task_id=todo.todo_id,
        )
    )
    refs.append(
        vfs.write_json(
            run_id,
            f"{research_prefix(todo)}/excerpts/result.json",
            {"excerpts": [item["excerpt"] for item in evidence_sources]},
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
                "summary": findings.summary,
                "sources": evidence_sources,
                "search_calls": outcome.search_calls,
                "search_budget": outcome.search_budget,
                "sufficient": findings.sufficient,
                "unresolved_gaps": findings.unresolved_gaps,
                "indexed_chunks": [
                    chunk.model_dump(mode="json", exclude={"text"}) for chunk in chunks
                ],
            },
            "evidence_bundle",
            task_id=todo.todo_id,
        )
    )
    return todo, refs, chunks


def run_worker(vfs, index, run_id, todo: ResearchTodo, researcher: ResearchAgent, *, thread_id=""):
    """Research one todo with its own sub-agent and persist the outcome."""
    if already_researched(vfs, run_id, todo):
        return restore_research(vfs, run_id, todo)
    outcome = researcher.research(todo.research_question, run_id=run_id, thread_id=thread_id)
    return persist_research(vfs, index, run_id, todo, outcome)
