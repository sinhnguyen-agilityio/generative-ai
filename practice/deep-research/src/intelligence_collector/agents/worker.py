"""Research task execution: run one research sub-agent and persist what it gathered.

The sub-agent (``researcher.py``) only searches and compresses. This module turns its
``ResearchOutcome`` into durable, run-scoped VFS artifacts and vector-index records.
"""

from __future__ import annotations

from intelligence_collector.agents.researcher import ResearchAgent, ResearchOutcome
from intelligence_collector.contracts import (
    IndexedChunk,
    ResearchGraphState,
    ResearchTodo,
    SourceRecord,
    Stage,
    TodoStatus,
)
from intelligence_collector.tools.url_tracking import UrlRegistry


def _evidence_path(todo: ResearchTodo) -> str:
    return f"research/{todo.todo_id}/evidence/result.json"


def already_researched(vfs, run_id: str, todo: ResearchTodo) -> bool:
    """Idempotent resume guard: an evidence bundle means the task finished earlier."""
    try:
        vfs.read(run_id, _evidence_path(todo))
    except FileNotFoundError:
        return False
    return True


def _write_failure(vfs, run_id: str, todo: ResearchTodo, message: str):
    return vfs.write_json(
        run_id,
        f"research/{todo.todo_id}/outcome.json",
        {"status": "failed", "question": todo.research_question, "error": message},
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
    """Write raw captures, notes, excerpts, and the evidence bundle; index raw content.

    Each URL is captured and indexed once per run: a URL already in the run's ``UrlRegistry``
    (from another task, gap research, or an earlier invocation) is referenced by its first
    capture instead of being written and indexed again. Pass a shared ``registry`` when
    persisting several tasks; otherwise it is loaded from and saved to the VFS here.

    A task is COMPLETED when it gathered sources and compressed findings, even if the findings
    report unresolved gaps (recorded in the evidence bundle for Verification). It is FAILED only
    when retrieval or compression produced nothing usable.
    """
    sources = outcome.unique_sources
    if not sources or outcome.findings is None:
        todo.status = TodoStatus.FAILED
        reason = "; ".join(outcome.errors) or "Research returned no sources"
        return todo, [_write_failure(vfs, run_id, todo, reason)], []

    urls = registry if registry is not None else UrlRegistry.load(vfs, run_id)
    refs, chunks, evidence_sources = [], [], []
    reused = 0
    for i, source in enumerate(sources):
        # Underscore keys (_query, _covered_by) are sub-agent bookkeeping, not page provenance.
        payload = {key: value for key, value in source.items() if not key.startswith("_")}
        record = SourceRecord(
            locator=source["url"],
            mcp_server=outcome.mcp_server,
            mcp_tool=outcome.mcp_tool,
            provenance={"query": source.get("_query", ""), **payload},
        )
        raw = source.get("content", "")
        entry = {
            "source_id": record.source_id,
            "locator": record.locator,
            "retrieved_at": record.retrieved_at.isoformat(),
            "mcp_server": record.mcp_server,
            "mcp_tool": record.mcp_tool,
            "provenance": record.provenance,
            "excerpt": raw,
        }
        known = urls.get(record.locator)
        if known is not None:
            # Already captured in this run: point at the first capture, skip write + index.
            reused += 1
            evidence_sources.append({
                **entry,
                "source_id": known["source_id"],
                "artifact_id": known["artifact_id"],
                "content_hash": known["content_hash"],
                "reused_from_task": known["task_id"],
            })
            continue
        ref = vfs.write_json(
            run_id,
            f"research/{todo.todo_id}/raw/{i}.json",
            {"source": record.model_dump(mode="json"), "content": raw},
            "raw_capture",
            task_id=todo.todo_id,
        )
        refs.append(ref)
        chunks.append(IndexedChunk(
            source_id=record.source_id,
            artifact_id=ref.artifact_id,
            text=raw,
            metadata={"run_id": run_id, "todo_id": todo.todo_id, "locator": record.locator},
        ))
        evidence_sources.append({
            **entry, "artifact_id": ref.artifact_id, "content_hash": ref.content_hash,
        })
        urls.register(record.locator, {
            "source_id": record.source_id,
            "artifact_id": ref.artifact_id,
            "content_hash": ref.content_hash,
            "task_id": todo.todo_id,
            "locator": record.locator,
        })
    if chunks:
        index.index(chunks)
    registry_ref = urls.save(vfs, run_id)
    if registry_ref is not None:
        refs.append(registry_ref)

    findings = outcome.findings
    todo.status = TodoStatus.COMPLETED
    refs.append(vfs.write(
        run_id,
        f"research/{todo.todo_id}/notes/result.md",
        f"# Research notes\n\nQuestion: {todo.research_question}\n\n"
        f"Search calls: {outcome.search_calls}/{outcome.search_budget}\n\n"
        f"Sources: {len(sources)} ({reused} already captured earlier in this run)\n\n"
        f"## Findings\n\n{findings.summary}\n",
        "notes",
        task_id=todo.todo_id,
    ))
    refs.append(vfs.write_json(
        run_id,
        f"research/{todo.todo_id}/excerpts/result.json",
        {"excerpts": [item["excerpt"] for item in evidence_sources]},
        "excerpts",
        task_id=todo.todo_id,
    ))
    refs.append(vfs.write_json(
        run_id,
        _evidence_path(todo),
        {
            "task_id": todo.todo_id,
            "question": todo.research_question,
            "summary": findings.summary,
            "sources": evidence_sources,
            "search_calls": outcome.search_calls,
            "search_budget": outcome.search_budget,
            "sufficient": findings.sufficient,
            "unresolved_gaps": findings.unresolved_gaps,
        },
        "evidence_bundle",
        task_id=todo.todo_id,
    ))
    return todo, refs, chunks


def run_worker(vfs, index, run_id, todo: ResearchTodo, researcher: ResearchAgent, *, thread_id=""):
    """Research one todo with its own sub-agent and persist the outcome."""
    if already_researched(vfs, run_id, todo):
        todo.status = TodoStatus.COMPLETED
        return todo, [], []
    outcome = researcher.research(todo.research_question, run_id=run_id, thread_id=thread_id)
    return persist_research(vfs, index, run_id, todo, outcome)


def targeted_research(vfs, index, run_id, question, researcher: ResearchAgent):
    todo = ResearchTodo(research_question=question)
    return run_worker(vfs, index, run_id, todo, researcher)


def targeted_research_node(
    state: ResearchGraphState,
    vfs,
    index,
    question: str,
    researcher: ResearchAgent,
) -> ResearchGraphState:
    """Research only the identified gap while retaining prior valid evidence."""
    if not question.strip():
        raise ValueError("targeted research requires a non-empty gap question")
    todo, refs, chunks = targeted_research(vfs, index, state.run_id, question, researcher)
    evidence_refs = [ref for ref in refs if ref.kind == "evidence_bundle"]
    return state.model_copy(update={
        "stage": Stage.SYNTHESIZING,
        "status": "active",
        "todo_ids": [*state.todo_ids, todo.todo_id],
        "todos": [*state.todos, todo],
        "artifacts": [*state.artifacts, *refs],
        "evidence": [*state.evidence, *evidence_refs],
        "index_refs": [*state.index_refs, *(chunk.chunk_id for chunk in chunks)],
        "progress_assessment": None,
        "verification": None,
        "approval": None,
        "publication": None,
        "last_error": "",
    })
