from __future__ import annotations

from typing import Any

from intelligence_collector.contracts import (
    IndexedChunk,
    ResearchGraphState,
    ResearchTodo,
    SourceRecord,
    Stage,
    TodoStatus,
)
from intelligence_collector.tools.mcp import ControlledMCP, MCPResearchPort


def _remaining_gaps(results: list[dict[str, Any]]) -> list[str]:
    gaps: list[str] = []
    for source in results:
        value = source.get("remaining_gaps", [])
        gaps.extend([value] if isinstance(value, str) else value)
    return [str(gap).strip() for gap in gaps if str(gap).strip()]


def _sufficient(results: list[dict[str, Any]]) -> bool:
    if not results:
        return False
    explicit = [source.get("sufficient") for source in results if "sufficient" in source]
    if explicit:
        return any(explicit)
    return any(str(source.get("content", "")).strip() for source in results)


def _next_query(question: str, results: list[dict[str, Any]], gaps: list[str]) -> str:
    for source in results:
        if str(source.get("next_query", "")).strip():
            return str(source["next_query"]).strip()
    return f"{question} {'; '.join(gaps)}" if gaps else f"{question} primary source details"


def _write_failure(vfs, run_id: str, todo: ResearchTodo, message: str):
    return vfs.write_json(
        run_id,
        f"research/{todo.todo_id}/outcome.json",
        {"status": "failed", "question": todo.research_question, "error": message},
        "research_outcome",
        task_id=todo.todo_id,
    )


def run_worker(
    vfs,
    index,
    run_id,
    todo: ResearchTodo,
    mcp: MCPResearchPort,
    *,
    complexity: str = "simple",
):
    existing = f"research/{todo.todo_id}/evidence/result.json"
    try:
        vfs.read(run_id, existing)
        todo.status = TodoStatus.COMPLETED
        return todo, [], []
    except FileNotFoundError:
        pass
    budget = min(5 if complexity == "complex" else 3, mcp.max_calls, 5)
    server_name = getattr(mcp, "server_name", "controlled")
    tool_name = getattr(mcp, "tool_name", "search")
    query = todo.research_question.strip()
    collected: list[dict[str, Any]] = []
    gaps: list[str] = []
    sufficient = False
    try:
        for _ in range(budget):
            result = mcp.search(query, server=server_name, tool=tool_name)
            collected.extend({**source, "_query": query} for source in result)
            gaps = _remaining_gaps(result)
            sufficient = _sufficient(result)
            if sufficient:
                break
            query = _next_query(todo.research_question, result, gaps)
    except Exception as exc:
        todo.status = TodoStatus.FAILED
        _write_failure(vfs, run_id, todo, f"{type(exc).__name__}: {exc}")
        raise RuntimeError(f"MCP retrieval failed for task {todo.todo_id}") from exc

    unique_sources = []
    seen = set()
    for source in collected:
        identity = (source.get("url"), source.get("content"))
        if identity not in seen:
            seen.add(identity)
            unique_sources.append(source)
    if not unique_sources:
        todo.status = TodoStatus.FAILED
        _write_failure(vfs, run_id, todo, "MCP returned no sources")
        raise RuntimeError("MCP returned no sources")

    chunks = []
    refs = []
    evidence_sources = []
    for i, source in enumerate(unique_sources):
        source_payload = {key: value for key, value in source.items() if key != "_query"}
        src = SourceRecord(
            locator=source["url"],
            mcp_server=server_name,
            mcp_tool=tool_name,
            provenance={"query": source.get("_query", query), **source_payload},
        )
        raw = source.get("content", "")
        ref = vfs.write_json(
            run_id,
            f"research/{todo.todo_id}/raw/{i}.json",
            {"source": src.model_dump(mode="json"), "content": raw},
            "raw_capture",
            task_id=todo.todo_id,
        )
        refs.append(ref)
        chunks.append(IndexedChunk(
            source_id=src.source_id,
            artifact_id=ref.artifact_id,
            text=raw,
            metadata={"run_id": run_id, "todo_id": todo.todo_id, "locator": src.locator},
        ))
        evidence_sources.append({
            "source_id": src.source_id,
            "locator": src.locator,
            "retrieved_at": src.retrieved_at.isoformat(),
            "mcp_server": src.mcp_server,
            "mcp_tool": src.mcp_tool,
            "provenance": src.provenance,
            "artifact_id": ref.artifact_id,
            "content_hash": ref.content_hash,
            "excerpt": raw,
        })
    index.index(chunks)
    todo.status = TodoStatus.COMPLETED if sufficient else TodoStatus.FAILED
    refs.append(vfs.write(
        run_id,
        f"research/{todo.todo_id}/notes/result.md",
        f"# Research notes\n\nQuestion: {todo.research_question}\n\n"
        f"Search calls: {mcp.calls}/{budget}\n\nSources: {len(unique_sources)}\n",
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
        f"research/{todo.todo_id}/evidence/result.json",
        {
            "task_id": todo.todo_id,
            "question": todo.research_question,
            "sources": evidence_sources,
            "search_calls": mcp.calls,
            "sufficient": sufficient,
            "unresolved_gaps": [] if sufficient else (gaps or ["Evidence remained insufficient"]),
        },
        "evidence_bundle",
        task_id=todo.todo_id,
    ))
    return todo, refs, chunks


def targeted_research(vfs, index, run_id, question, mcp):
    todo = ResearchTodo(research_question=question)
    return run_worker(vfs, index, run_id, todo, mcp)


def targeted_research_node(
    state: ResearchGraphState,
    vfs,
    index,
    question: str,
    mcp: MCPResearchPort,
) -> ResearchGraphState:
    """Research only the identified gap while retaining prior valid evidence."""
    if not question.strip():
        raise ValueError("targeted research requires a non-empty gap question")
    todo, refs, chunks = targeted_research(vfs, index, state.run_id, question, mcp)
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
