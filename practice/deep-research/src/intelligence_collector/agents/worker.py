from intelligence_collector.contracts import IndexedChunk, ResearchTodo, SourceRecord, TodoStatus
class ControlledMCP:
    def __init__(self, sources): self.sources, self.calls = sources, 0
    def search(self, query): self.calls += 1; return self.sources
def run_worker(vfs, index, run_id, todo: ResearchTodo, mcp: ControlledMCP):
    result = mcp.search(todo.research_question)
    if not result: raise RuntimeError("MCP returned no sources")
    chunks=[]; refs=[]
    for i, source in enumerate(result):
        src = SourceRecord(locator=source["url"], provenance=source)
        raw = source.get("content", "")
        ref = vfs.write_json(run_id, f"research/{todo.todo_id}/raw/{i}.json", {"source": src.model_dump(mode="json"), "content": raw}, "raw_capture", task_id=todo.todo_id)
        refs.append(ref); chunks.append(IndexedChunk(source_id=src.source_id, artifact_id=ref.artifact_id, text=raw, metadata={"todo_id": todo.todo_id}))
    index.index(chunks); todo.status = TodoStatus.COMPLETED
    vfs.write_json(run_id, f"research/{todo.todo_id}/evidence/result.json", {"artifact_ids": [r.artifact_id for r in refs]}, "evidence_bundle", task_id=todo.todo_id)
    return todo, refs, chunks

