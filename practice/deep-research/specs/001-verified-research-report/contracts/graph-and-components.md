# Graph and Component Contracts

## Node contract

Each node accepts an immutable state snapshot plus injected ports and returns a typed partial state
update or a typed `NodeFailure`. Nodes may write only through injected repositories and must emit
an artifact reference for each required stage write.

## Required nodes and edges

`guard_input → clarify_with_user → write_research_brief → planning → supervisor → research_worker → research_complete_check →
synthesis → verify → bvf_router → (revise_draft | targeted_research | replan | approval_interrupt) →
publish`.

`clarify_with_user` conditionally ends the current invocation after appending one assistant question,
or routes a validated clear decision to `write_research_brief`. That node returns a `ResearchBrief`
containing one detailed research question and
routes to `planning` without a user approval edge. `planning` must persist todos via `write_todos`
before any retrieval-capable successor is enabled.

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

## Langfuse trace contract

Each live graph invocation creates exactly one root `agent` observation. Its Langfuse session ID is
the LangGraph `thread_id`; metadata includes `run_id` and a unique `execution_id`. Graph nodes are
child spans named `graph.<node>`. LangChain model calls are nested generation observations, while
structured parsing, retries, selected model, fallback, prompt version, and safe failures are recorded
as child validation spans. The client flushes before the invocation returns. Trace payloads exclude
API keys, Langfuse credentials, and environment-variable values.
