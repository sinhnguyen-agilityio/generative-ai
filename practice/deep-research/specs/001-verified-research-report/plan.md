# Implementation Plan: Verified Research Report

**Branch**: `001-verified-research-report` | **Date**: 2026-09-24 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-verified-research-report/spec.md`

## Summary

Build a local Python application with a Streamlit chat UI and a directly composed LangGraph
workflow. The graph validates and guards requests, conducts multi-turn scope clarification,
automatically enters Planning when scope is bounded, writes todos before retrieval, supervises
isolated MCP research workers, persists stage artifacts through a run-scoped VFS, indexes raw
source data in a local vector store, synthesizes and verifies evidence-linked reports, routes
typed verification feedback through bounded correction loops, and pauses for human approval before
atomic publication. Langfuse tracing and repeatable Ragas faithfulness evaluation provide
observability and quality evidence. The Deep Agents SDK is explicitly excluded.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**: Streamlit; LangGraph directly; LangChain chat models and prompt/message
interfaces; `langchain.chat_models.init_chat_model`; Pydantic v2; an MCP client; a local vector
store; Langfuse SDK; Ragas; pytest. The Deep Agents SDK is prohibited.

**Storage**: Run-scoped VFS directory for immutable/versioned artifacts; SQLite (or equivalent
local checkpoint store) for LangGraph checkpoints and run metadata; local vector index for raw
research retrieval; user-configured permanent output directory written only by publication.

**Testing**: pytest unit and contract tests; integration tests with deterministic fake models,
fake MCP servers, fake vector store, and temporary VFS; Streamlit AppTest for UI behavior; repeatable
Ragas evaluation dataset for faithfulness.

**Target Platform**: Local developer workstation running Python and Streamlit; network access only
through configured MCP sources and optional Langfuse endpoint.

**Project Type**: Local web application with a Streamlit UI and application services.

**Performance Goals**: Scope responses and stage-status refreshes visible within 2 seconds excluding
model/tool latency; no unbounded polling; resume restores a run from its latest valid checkpoint
without redoing completed retrieval; search-call budgets are enforced exactly.

**Constraints**: No Deep Agents SDK; LangGraph owns all transitions and checkpoints; no retrieval
before durable `write_todos`; no authoritative state in chat history; raw source data and all stage
artifacts remain run-scoped; max five searches per task; bounded recovery budget; guardrails reject
prompt injection and disallowed offensive/violent requests before workflow mutation; no permanent
output write without human approval bound to the unchanged verified artifact.

**LLM Decision Contract**: Live semantic roles use `ModelRouter.invoke_structured()` with explicit
versioned prompts and Pydantic output schemas. `mode="live"` requires a configured primary model;
`mode="fake"` is explicit and test-only. Router traces role, prompt version, selected model,
fallback/error events, and validated decision metadata without secrets.

**LangChain Chat Architecture**: `ModelFactory` initializes each configured live model with
`langchain.chat_models.init_chat_model`; the router binds role-specific primary/fallback models,
timeouts, and provider reasoning settings. Role prompts are `ChatPromptTemplate` objects composed
of `SystemMessage`, `MessagesPlaceholder`, and data variables. Scope history is stored in LangGraph
state under the thread ID and passed as `HumanMessage`/`AIMessage` turns. Invalid structured output
is retried within bounds or recorded as an error.

**Langfuse Trace Hierarchy**: A live Streamlit submission creates one root `agent` observation with
a deterministic execution trace ID. Langfuse `session_id` equals the LangGraph `thread_id`, while
`run_id` and a per-submission `execution_id` are propagated metadata. Every graph node creates a
child span, LangChain's callback handler records nested model generations and token usage, and the
router records structured validation/fallback metadata. The client flushes after every execution.
Live execution fails before model invocation when tracing is disabled or misconfigured; offline
tests use a recording client and never require or expose credentials.

**Scale/Scope**: One local user and one active run at a time for the first release; dozens of
research tasks and thousands of source chunks per run; local persistence is the source of truth.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- Validated Human Input: PASS — input schema validation and guardrail rejection precede state
  mutation, tool calls, and artifact writes.
- Explicit, Resumable Workflow: PASS — LangGraph is the only transition owner; typed state,
  conditional edges, bounded loops, and durable checkpoints are specified.
- Durable Research Artifacts: PASS — VFS stage artifacts, checkpoint storage, and vector index are
  separate from conversation history.
- Evidence-Linked Research: PASS — MCP provenance, immutable evidence bundles, claim references,
  and vector retrieval metadata are required.
- Verified, Human-Approved Publication: PASS — typed verification precedes an interrupt; publication
  checks the approved artifact hash and writes atomically.
- Delivery and Review Gates: PASS — component contracts, failure behavior, acceptance flows, tests,
  tracing, and evaluation are defined in this plan and its design artifacts.

## Phase 0: Research Decisions

See [research.md](./research.md). Decisions resolve all implementation unknowns without introducing
alternatives that would violate the constitution or clarified specification.

## Phase 1: Design Artifacts

- [data-model.md](./data-model.md): typed state, persisted entities, lifecycle transitions, and
  artifact layout.
- [contracts/](./contracts/): component, graph, tool, VFS, verification, publication, and UI
  contracts.
- [quickstart.md](./quickstart.md): repeatable local demonstrations for end-to-end, resume,
  fallback, every BVF route, guardrails, and approval binding.

## End-to-End Execution Sequence

1. Streamlit accepts a chat turn, validates its shape, and runs the guardrail. Rejected input is
   displayed without mutating the run.
2. `clarify_with_user` adds the current `HumanMessage` to checkpointed conversation history and
   passes that actual history through `MessagesPlaceholder` to a structured Scope LLM decision. A
   clarification result appends exactly one focused `AIMessage` and ends the invocation in Scoping.
3. A clear result appends exactly one concise verification `AIMessage` and routes to
   `write_research_brief` without user approval. That node makes a
   second structured call with its own versioned prompt, validates a dedicated single-question
   `ResearchBrief`,
   writes `scope/brief_summary.v1.json`, and enters Planning without a brief approval interrupt.
4. Planning runs the Planner Agent: a structured `planner.v2` LLM call decomposes the brief's
   research question into 1–5 standalone, non-overlapping research questions (`TodoPlan.questions`).
   The system turns each question into a `planned` todo and persists the list via `write_todos`.
   A non-empty, valid, durably stored todo list is required before any Research/retrieval edge.
5. The Supervisor assigns isolated workers. Each worker uses MCP only, starts broad, assesses gaps
   after each call, narrows as needed, and stops early or at the five-call hard limit. Raw source
   captures, excerpts, notes, evidence bundles, and vector-index records are written to the VFS.
6. `think_tool` creates a progress assessment. The Supervisor may issue `ResearchComplete` only if
   every todo is completed and all VFS/index prerequisites are present. This ends Research only.
7. Synthesis retrieves indexed chunks and evidence bundles, writes previous and latest draft
   artifacts, and emits a versioned candidate report.
8. Verification returns pass or one typed issue: `DRAFT_DEFECT`, `EVIDENCE_GAP`, or `PLAN_GAP`.
   BVF routing sends each issue to draft revision, targeted research, or re-planning respectively;
   valid evidence is retained and a progress fingerprint prevents no-op loops.
9. A finite recovery counter is decremented for each repair. Exhaustion produces `NEEDS_REVIEW`.
10. A passing result enters `Awaiting Approval` and LangGraph interrupts. The user approves or
    rejects the exact artifact hash. Approval resumes the graph; rejection records the decision and
    leaves permanent output unchanged.
11. Publication verifies the artifact hash and verification status, writes the permanent artifact
    atomically, records a Publication Record, and transitions to Published.

## Component Responsibilities and Boundaries

| Component | Responsibility | Contract / failure behavior | Integration points |
|---|---|---|---|
| Streamlit UI | Chat, stage/progress display, report/evidence review, approval decision | Renders typed state; never writes authoritative artifacts; displays actionable errors | Graph runner, VFS read API |
| Input Validator + Guardrail | Validate shape/size and reject prompt injection or disallowed offensive/violent requests | Returns typed `GuardrailDecision`; rejection has no state mutation | Scope entry edge, audit event |
| Scope Agent | `clarify_with_user` evaluates actual message history; `write_research_brief` creates a dedicated validated `ResearchBrief` after a clear decision | One-question clarification or complete brief; model/validation/persistence failure cannot become clear | LangChain chat model/router, LangGraph thread state, checkpoint, brief VFS artifact |
| Planner Agent + `write_todos` | `PlannerAgent.define_questions` turns the `ResearchBrief` into a list of research questions; `PlannerAgent.write_todos` converts them to `planned` todos and persists them | `TodoPlan` (questions only, 1–5, non-blank, unique); invalid output or failed write blocks Research | Model router (`planner` role), VFS, graph state |
| Research Supervisor | Assign workers, aggregate outcomes, use `think_tool`, assess readiness | `ProgressAssessment`; cannot ResearchComplete until all todos complete and artifacts indexed | Worker pool, VFS, vector store |
| MCP Research Worker | Search external sources, capture provenance, create evidence | Per-task search budget; MCP failures are task outcomes; isolated context | MCP clients, VFS, vector store, model router |
| VFS Artifact Store | Atomic run-scoped files and immutable versions | Reject path escape/cross-run writes; partial writes are invisible | All nodes, checkpoint metadata |
| Vector Index | Chunk/index raw source captures and retrieve synthesis context | Index failures block ResearchComplete; retrieval records source IDs | Worker, synthesis, verifier |
| Synthesizer | Draft report from indexed evidence and scope | Missing evidence is a draft/verification issue, not invented content | Model router, VFS, vector store |
| Verifier | Scope coverage, claim/evidence links, typed feedback | Exactly one primary issue type on failure; pass enables approval | Evidence store, graph conditional edges |
| Model Router | Role-specific primary/fallback models and cost-conscious reasoning sandwich | Retries bounded; fallback event traced; final failure is visible | All model-using nodes, Langfuse |
| Checkpointer | Persist typed graph state and resume points | Atomic checkpoint; invalid latest checkpoint falls back or NEEDS_REVIEW | LangGraph, VFS metadata |
| Approval/Publisher | Bind approval to artifact hash and atomically publish | Reject/stale/unverified writes nothing permanent | Streamlit interrupt, VFS, output path |
| Tracing/Evaluation | Create root execution traces grouped by thread, nested node spans, model generations, validation/fallback events, and evaluate faithfulness | Missing live tracing config blocks execution; export/flush failure is surfaced without exposing credentials | Langfuse v4 SDK and LangChain callback, Ragas dataset |

## Project Structure

```text
src/intelligence_collector/
├── app.py
├── config.py
├── contracts.py
├── graph/
│   ├── state.py
│   ├── build.py            # wiring only: nodes + edges per topology
│   ├── nodes.py            # Node names, ScopeNodes, PlannerNodes, ResearchNodes, approval, stubs
│   ├── routing.py          # named conditional-edge routers + BVF routing
│   └── instrumentation.py  # node_wrapper: lifecycle, checkpoint, tracing
├── agents/
│   ├── scope.py
│   ├── planner.py
│   ├── supervisor.py
│   ├── worker.py
│   ├── synthesis.py
│   └── verification.py
├── tools/
│   ├── write_todos.py
│   ├── think_tool.py
│   └── mcp.py
├── storage/
│   ├── vfs.py
│   ├── checkpoints.py
│   ├── artifacts.py
│   └── vector_index.py
├── models/router.py
├── guardrails.py
├── publication.py
└── observability/
    ├── tracing.py
    └── evaluation.py

tests/
├── unit/
├── contract/
├── integration/
├── evaluation/
└── ui/
```

**Structure Decision**: A single local Python application keeps the first implementation simple;
domain services are separated from graph wiring, storage, integrations, and UI so contracts can be
tested independently and no UI path can bypass workflow gates.

### Graph Module Organization (FR-030a)

- `graph/nodes.py` owns the `Node` name constants (single source for wiring, `app.py` activity
  labels, and `graph.<node>` trace spans) and all node logic. Collaborator-dependent nodes are
  grouped into injected classes: `ScopeNodes(scope_agent, vfs)`, `PlannerNodes(planner_agent, vfs)`,
  and `ResearchNodes(vfs, index, mcp)`. `build_graph(planner_agent=...)` defaults to a
  `PlannerAgent` sharing the scope agent's model router.
- `graph/routing.py` holds named routers for every conditional edge: `route_after_clarify`
  (END when clarification is needed, else `write_research_brief`) and `route_after_verify`
  (`approval` only when `verification.passed`; a `None`/missing result ends the run).
- `graph/instrumentation.py` provides `node_wrapper`, applied uniformly to every node for lifecycle
  events, checkpoint load/save, and Langfuse spans.
- `graph/build.py` contains only wiring. `build_graph` supports three topologies selected by
  injected collaborators:
  - Full (`scope_agent` + `research_nodes`): `clarify_with_user` → (END | `write_research_brief`) →
    `planning` → `researching` → `research_complete` → `synthesizing` → `verifying` →
    (`approval` when passed and `with_approval` | END).
  - Scope only (`scope_agent`): … → `planning` → END.
  - Stage-only stubs (neither): `scope` → `planning` → `researching` → `synthesizing` →
    `verifying` → [`approval`] → END, used by contract tests.

## Constitution Check (Post-Design)

- All gates remain PASS. The design has no Deep Agents SDK, no unbounded route, no chat-history
  authority, and no publication path that bypasses verification and approval.
- Complexity is justified only by explicit requirements: graph/checkpoints, VFS/vector provenance,
  typed BVF recovery, MCP isolation, model fallback, and observability/evaluation.

## Complexity Tracking

| Addition | Why needed | Simpler alternative rejected because |
|---|---|---|
| Typed LangGraph workflow with checkpointing | Required resumability, bounded recovery, and approval interrupt | Linear script cannot safely resume or enforce conditional gates |
| VFS plus vector index | Required durable stage evidence and synthesis retrieval | Chat history cannot provide authoritative provenance or scalable retrieval |
| Supervisor/workers and typed BVF routing | Required isolated MCP research and deterministic repair behavior | One monolithic agent cannot expose task progress or route typed verification issues |
| Model router, Langfuse, and Ragas | Required fallback, cost control, tracing, and repeatable faithfulness evaluation | Single model/no telemetry cannot demonstrate resilience or quality |
