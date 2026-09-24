# Phase 0 Research Decisions

## Decision: Direct LangGraph composition

Use LangGraph StateGraph/compiled graphs directly with typed state and explicit node functions.
Do not use the Deep Agents SDK. LangGraph is the single owner of transitions, conditional routing,
interrupt/resume, and checkpoint identifiers.

Rationale: The specification requires observable transitions, bounded recovery, and resumable
execution. Direct composition keeps those contracts visible and testable.

Alternatives considered: A monolithic agent loop was rejected because it hides state transitions;
Deep Agents SDK was rejected by explicit user constraint.

## Decision: Local durable storage boundaries

Use a run-scoped VFS rooted outside conversation state, a local checkpoint database, and a local
vector index. Files are immutable by version and written atomically through a path-safe repository.

Rationale: Separate authorities preserve provenance and allow resume without replaying retrieval.

Alternatives considered: Chat/session memory was rejected by the constitution; a remote database is
deferred because the first release is local and single-user.

## Decision: Typed contracts at every boundary

Use Pydantic models for graph state slices, todos, MCP results, artifacts, verification issues,
approval decisions, and publication records. Serialize stable IDs, run IDs, artifact hashes, and
schema versions.

Rationale: Typed validation makes invalid transitions and stale approvals detectable before writes.

## Decision: Research budgets and completion gate

Workers use broad-to-narrow MCP searches, stop when confident, use at most three calls for simple
questions and five calls absolutely for complex questions. `ResearchComplete` is a validated request
that requires every todo completed, VFS artifacts present, and vector indexing successful.

Rationale: Explicit budgets prevent runaway cost and the strict gate prevents synthesis from hiding
unfinished work.

## Decision: BVF issue routing

Verification emits one primary issue type. `DRAFT_DEFECT` routes to draft revision,
`EVIDENCE_GAP` to targeted research, and `PLAN_GAP` to affected-todo re-planning. Each repair must
change a progress fingerprint (artifact hash, evidence set, or task plan) or the run stops as
`NEEDS_REVIEW`; recovery attempts are finite.

## Decision: Guardrail placement

Input validation and guardrails run before graph mutation and before any tool or model call that
could persist or retrieve research. The guardrail rejects prompt-injection instructions and
requests that are disallowed offensive or violent content, returning a safe actionable message.

Rationale: Early rejection protects tools, artifacts, credentials, and downstream prompts.

## Decision: Cost-conscious reasoning sandwich

The model router supports role-specific primary and fallback models. Use a cheaper model for scope
normalization, planning, progress reflection, and draft repair; use the configured stronger model for
synthesis and verification; invoke fallback only on bounded retryable failures. Every selection and
fallback is traced.

## Decision: Observability and evaluation

Trace run, graph node, MCP call, model selection, artifact write, verification, interrupt, and
publication events in Langfuse when configured. A fixed local Ragas dataset evaluates faithfulness
against stored contexts; evaluation is repeatable and never changes publication status by itself.
