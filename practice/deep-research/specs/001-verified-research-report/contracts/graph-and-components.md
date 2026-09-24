# Graph and Component Contracts

## Node contract

Each node accepts an immutable state snapshot plus injected ports and returns a typed partial state
update or a typed `NodeFailure`. Nodes may write only through injected repositories and must emit
an artifact reference for each required stage write.

## Required nodes and edges

`guard_input → scope → planner → supervisor → research_worker → research_complete_check →
synthesis → verify → bvf_router → (revise_draft | targeted_research | replan | approval_interrupt) →
publish`.

Conditional edges must inspect typed state, not UI text. `research_complete_check` rejects incomplete
todos or missing VFS/index refs. `bvf_router` rejects unknown issue types and no-progress fingerprints.

## Port interfaces

- `ModelPort.invoke(role, messages, schema) -> StructuredModelResult`
- `MCPPort.search(server, query, limits) -> SourceResult[]`
- `VFSArtifactPort.write(run_id, kind, content, metadata) -> ArtifactRef`
- `VectorIndexPort.index(chunks) -> IndexRef`; `retrieve(query, filters) -> IndexedChunk[]`
- `CheckpointPort.save/load(thread_id) -> Checkpoint`
- `TracePort.span(event, attributes) -> TraceRef`

Every port failure is typed, includes retryability, and preserves already committed artifacts.
