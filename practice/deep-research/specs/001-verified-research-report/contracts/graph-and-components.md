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

Each research run has exactly one trace, whose ID is derived from `run_id`, so any process can
attach later work to it; its Langfuse session ID is the LangGraph `thread_id`. Every graph
invocation (start, resume after an interrupt, recovery, continuation) creates one
execution observation (metadata `execution_kind`, `interrupt_kind`, `execution_id`, `agents`),
named `<kind>[ · <interrupt_kind>]: <agent> → … ⏸ <paused_at>`, that is closed when the
invocation returns; no observation stays open across a human pause.
The Langfuse LangChain callback created for the execution is passed in the graph config, so
node runs (named `<agent> · <node>` from `NODE_AGENT`), research sub-agent runs
(`research_agent · <todo_id>`) and model generations
nest beneath it. MCP calls (`mcp.search`), persistence (`research.persist`), vector-index
operations and RAGAS evaluators are child observations of the current run. Each LLM call is
recorded once. Exported payloads are masked (credentials, secret URL parameters, strings over
4,000 characters). Tracing failures never change workflow behaviour. Details:
`docs/observability.md`.
