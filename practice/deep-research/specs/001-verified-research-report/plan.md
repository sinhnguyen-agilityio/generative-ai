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
atomic publication. Langfuse tracing and RAGAS Faithfulness scoring of each passing report (LLM
judge, informational only) provide observability and quality evidence. The Deep Agents SDK is
explicitly excluded.

## Technical Context

**Language/Version**: Python 3.11+

**Primary Dependencies**: Streamlit; LangGraph directly; LangChain chat models and prompt/message
interfaces; `langchain.chat_models.init_chat_model`; Pydantic v2; an MCP client; a local vector
store; Langfuse SDK; Ragas; pytest. The Deep Agents SDK is prohibited.

**Storage**: Run-scoped VFS directory for immutable/versioned artifacts; SQLite (or equivalent
local checkpoint store) for LangGraph checkpoints and run metadata; local vector index for raw
research retrieval; user-configured permanent output directory written only by publication.

**Testing**: pytest unit and contract tests; integration tests with deterministic fake models,
fake MCP servers, fake vector store, and temporary VFS; Streamlit AppTest for UI behavior; a
deterministic local word-overlap baseline plus RAGAS's own `Faithfulness` metric (judge-LLM calls
mocked in unit tests; the real Gemini judge — `gemini-3.7-flash` by default,
`INTELLIGENCE_FAITHFULNESS_MODEL` — only runs live, via `app._get_scope_graph`'s injected
evaluator).

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

1. Streamlit accepts a chat turn, normalizes it (`guardrails.normalize_text`: Unicode/confusable
   folding, invisible-character stripping) to close homoglyph evasion, then validates its
   shape/size/regex guardrail. Rejected input is displayed without mutating the run.
2. `clarify_with_user` first runs a pre-model content-safety guardrail (`ScopeAgent
   .check_content_safety`, `guardrail` role, standard tier): a blocked message raises
   `GuardrailViolation` and never reaches the Scope model. An allowed message is added as the current
   `HumanMessage` to checkpointed conversation history and
   passes that actual history through `MessagesPlaceholder` to a structured Scope LLM decision. A
   clarification result appends exactly one focused `AIMessage` and ends the invocation in Scoping.
3. A clear result appends exactly one concise verification `AIMessage` and routes to
   `write_research_brief` without user approval. That node makes a
   second structured call with its own versioned prompt, validates a dedicated single-question
   `ResearchBrief`,
   writes `scope/brief_summary.v1.json`, and enters Planning without a brief approval interrupt.
4. Planning runs the Planner Agent: a structured `planner.v3` LLM call decomposes the brief's
   research question into 1–5 standalone, non-overlapping research questions (`TodoPlan.questions`)
   and 1–5 acceptance criteria (`TodoPlan.acceptance_criteria`) describing what a finished report
   must state to count as achieving the brief's goal. The system turns each question into a
   `planned` todo, persists the list via `write_todos`, and persists the acceptance criteria to
   `planning/acceptance_criteria.v1.json` and graph state. A non-empty, valid, durably stored todo
   list is required before any Research/retrieval edge.
5. The Research Supervisor delegates each planned todo to its own research sub-agent, running up to
   `INTELLIGENCE_MAX_CONCURRENT_RESEARCHERS` (default 3) in parallel. Each sub-agent gets a fresh
   Tavily MCP client and runs the loop `llm_call` → (`tool_node` → `llm_call`)* →
   `compress_research`: the LLM (`researcher.v1`) starts broad, assesses gaps after each
   `tavily_search` result, narrows as needed, and stops with a final note or at the per-task
   five-call hard limit; compression (`research-compress.v1`) returns cited `ResearchFindings`.
   After all sub-agents return, the Supervisor writes raw source captures, notes (with findings),
   excerpts, evidence bundles, and vector-index records to the VFS in plan order. URLs are tracked
   by normalized identity: a sub-agent hides pages it already retrieved from later search results
   (`researcher.v2`), and the run-level `research/url_registry.json` makes later tasks, gap
   research, and resumed runs reference the first capture instead of re-capturing and re-indexing.
6. `think_tool` creates a progress assessment with explicit blockers. The Supervisor may issue
   `ResearchComplete` only if every todo is completed and all VFS/index prerequisites are present.
   This ends Research only.
7. **Build**: for each research question, `select_relevant_evidence` retrieves its top-K most
   relevant sources (`INTELLIGENCE_EVIDENCE_TOP_K`, default 6) from the Chroma vector index —
   Gemini embeddings by default (`INTELLIGENCE_EMBEDDING_MODEL`, default `gemini-embedding-2`;
   `INTELLIGENCE_EMBEDDING_PROVIDER`, default `google_genai`; set to `openai` with an OpenAI
   model name to switch providers), persisted
   locally under `INTELLIGENCE_CHROMA_PATH` — scoped to that question so no question's sources can
   be crowded out by another's; a bundle at or under top-K, or a retrieval failure, is left
   unfiltered. The Report Writer (`report-writer.v1`, structured LLM call) then drafts the report
   from that narrowed evidence, citing the (unchanged, still-complete) per-run source catalog as
   `[S<n>]`, and persists `reports/draft.v<n>.md`, `reports/previous.md`, and `reports/latest.md`.
8. **Verify**: the deterministic source-verification script (`source_verification.py`, also runnable
   as `python -m intelligence_collector.source_verification --run-id RUN`) runs ten checks on the
   report text against the stored sources and returns pass or one primary issue: `EVIDENCE_GAP`
   (no evidence, or unresolved research gaps not yet researched) or `DRAFT_DEFECT` (structure,
   unknown citation ids, uncited claims, ungrounded numbers/wording, invented URLs, Sources list,
   uncovered questions, undisclosed gaps). Only if that script passes, `AcceptanceVerifier`
   (`acceptance.v1`, role `verify`) judges the report against each planner-defined acceptance
   criterion; any unmet criterion is folded in as an additional `acceptance_criteria` check and
   turns the result into a failed `DRAFT_DEFECT`. Only a report passing every check, including
   acceptance, leaves the loop.
9. **Fix** (self-improving): each failure increments the visible attempt counter. For a
   `DRAFT_DEFECT`, a reflection call (`bvf-reflect.v1`) diagnoses the root cause, adds reusable
   lessons to an accumulating lesson memory, and plans a strategy; the next Build receives the
   failed checks, all lessons, the strategy, and the previous draft. For an `EVIDENCE_GAP`, one
   targeted research sub-agent runs per unresearched gap, then the report is rebuilt and must
   disclose remaining gaps under `## Limitations`. A failure fingerprint detects no progress: the
   first repeat tells the reflection to change approach; two consecutive repeats, or reaching
   `INTELLIGENCE_RECOVERY_ATTEMPTS` (default 3), end in `needs_review` (`NEEDS_REVIEW`). The script
   never emits `PLAN_GAP`; the router sends it to `needs_review` if it ever occurs.
10. A passing result enters `Awaiting Approval` and LangGraph interrupts. The user approves or
    rejects the exact artifact hash. Approval resumes the graph; rejection records the decision and
    leaves permanent output unchanged.
11. Publication verifies the artifact hash and verification status, writes the permanent artifact
    atomically, records a Publication Record, and transitions to Published.

## Component Responsibilities and Boundaries

| Component | Responsibility | Contract / failure behavior | Integration points |
|---|---|---|---|
| Streamlit UI | Chat, stage/progress display, report/evidence review, approval decision, "New thread" reset, one live-updating row per pipeline agent (content left, status badge right) driven by `graph.stream(stream_mode="values")` (`app.py::_render_agent_row`, `_render_agent_pipeline`) | Renders typed state; never writes authoritative artifacts; displays actionable errors; agent status derived from `activity` lifecycle entries grouped by agent name; a row stays hidden (placeholder cleared) until that agent's status leaves `pending`, so the pipeline never lists every agent up front; rows redraw into `st.empty()` placeholders on every streamed step, not only once after the whole run | Graph runner (`.stream()`, not `.invoke()`), VFS read API (reads each completed todo's evidence bundle for its source links) |
| Input Normalizer + Guardrail (`guardrails.py`) | Fold homoglyphs/compatibility forms and strip invisible characters (`normalize_text`); validate shape/size and reject prompt injection or disallowed offensive/violent requests via regex | Returns typed `GuardrailDecision`; rejection has no state mutation | Scope entry edge, audit event |
| Scope Agent | `check_content_safety` runs a pre-model LLM guardrail (`guardrail` role, standard tier) that blocks vulgar/prohibited messages before any Scope call; `clarify_with_user` evaluates actual message history; `write_research_brief` creates a dedicated validated `ResearchBrief` after a clear decision | Blocked content raises `GuardrailViolation` with zero Scope-model calls; otherwise one-question clarification or complete brief; model/validation/persistence failure cannot become clear | Model router (`guardrail` role, standard tier; `scope` role, reasoning tier), LangGraph thread state, checkpoint, brief VFS artifact |
| Planner Agent + `write_todos` | `PlannerAgent.define_plan` turns the `ResearchBrief` into research questions and acceptance criteria; `PlannerAgent.write_todos` converts the questions to `planned` todos, persists them, and persists the criteria | `TodoPlan` (questions and acceptance_criteria, 1–5 each, non-blank, unique); invalid output or failed write blocks Research | Model router (`planner` role, reasoning tier), VFS, graph state |
| Acceptance Verifier (`agents/acceptance_verifier.py`) | Judge the passing report against each planner-defined acceptance criterion, only after source-verification passes | `AcceptanceVerdict` (met/unmet + justification per criterion); any unmet criterion folds into `VerificationResult` as a `DRAFT_DEFECT` | Model router (`verify` role, reasoning tier), graph state (`acceptance_criteria`) |
| Research Supervisor (`agents/supervisor.py`) | Delegate one sub-agent per todo (bounded thread pool), persist outcomes in plan order, use `think_tool`, assess readiness | `ProgressAssessment` with blockers; a failing sub-agent becomes a FAILED task outcome, never cancels siblings; cannot ResearchComplete until all todos complete and artifacts indexed | `ResearchAgent`, VFS, vector store |
| Research Sub-agent (`agents/researcher.py`) | LangGraph loop: LLM decision → `tavily_search` tool execution → routing → research compression | Fresh MCP client and hard per-task budget (≤5); MCP failures become tool/task outcomes with redacted errors; `ResearchFindings` validated | Model router (`researcher`, `research_compressor` roles, tool calling), Tavily MCP |
| Research persistence (`agents/worker.py`) | Turn a `ResearchOutcome` into raw captures, notes, excerpts, evidence bundle, and index chunks; idempotent resume guard; capture each URL once per run | COMPLETED with sources + findings (gaps recorded); FAILED writes `outcome.json`; already-captured URLs reference the first capture (`reused_from_task`) | VFS, vector store, `UrlRegistry` |
| VFS Artifact Store | Atomic run-scoped files and immutable versions | Reject path escape/cross-run writes; partial writes are invisible | All nodes, checkpoint metadata |
| Vector Index | Chunk/index raw source captures and retrieve synthesis context | Index failures block ResearchComplete; retrieval records source IDs | Worker, synthesis, verifier |
| Report Writer (`agents/report_writer.py`) | Build/Fix drafts from top-K retrieved evidence (`select_relevant_evidence`, per question) with `[S<n>]` citations; BVF reflection producing lessons and strategy | `DraftAssessment` / `Reflection` validated; fix attempts must receive failures, lessons, strategy, previous draft; retrieval scoped per question, never global | Model router (`synthesizer`, `fixer` roles), VFS, vector index |
| Vector Index (`storage/vector_index.py`, `storage/chroma_vector_index.py`) | `LocalVectorIndex` (lexical, offline/tests) and `ChromaVectorIndex` (Gemini embeddings by default, OpenAI selectable via `INTELLIGENCE_EMBEDDING_PROVIDER`, local Chroma persistence, live) behind one `VectorIndex` interface | `retrieve(query, limit, run_id, todo_id)`; Verify never reads through this — it always loads the full evidence catalog from VFS independently | Gemini/OpenAI embeddings API, local Chroma directory |
| Source Verifier (`source_verification.py`) | Deterministic source-verification script over report text and evidence bundles; standalone CLI | Ten checks; exactly one primary issue type on failure; failure fingerprint for no-progress detection; pass enables approval | Evidence bundles in VFS, `verification/catalog_state.json`, graph conditional edges |
| Model Router (`models/router.py`, `models/role_router.py`) | `RoleModelRouter` dispatches each role to one of two profiles: reasoning (`scope`, `planner`, `verify`) or standard (`researcher`, `research_compressor`, `synthesizer`, `fixer`) — the "Reasoning Sandwich": a high-effort model plans and verifies around standard-tier implementation work. Live app default: Gemini for both tiers (`INTELLIGENCE_REASONING_MODEL/PROVIDER` = `gemini-3.8-flash`/`google_genai` at `reasoning_effort=high`, the ceiling `ChatGoogleGenerativeAI` accepts — neither Gemini nor OpenAI's SDK has an "xhigh" literal, that word is only this tier's name; `INTELLIGENCE_STANDARD_MODEL/PROVIDER` = `gemini-3.5-flash-lite`/`google_genai` at `reasoning_effort=low`), each with a Groq fallback (`openai/gpt-oss-120b` for reasoning, `openai/gpt-oss-20b` for standard), built once via `app._get_role_router` and shared by every agent | Retries bounded per profile; fallback event traced; final failure is visible | All model-using nodes, Langfuse |
| Checkpointer | Persist typed graph state and resume points | Atomic checkpoint; invalid latest checkpoint falls back or NEEDS_REVIEW | LangGraph, VFS metadata |
| Approval/Publisher | Bind approval to artifact hash and atomically publish | Reject/stale/unverified writes nothing permanent | Streamlit interrupt, VFS, output path |
| Tracing/Evaluation | Create root execution traces grouped by thread, nested node spans, model generations, and validation/fallback events | Missing live tracing config blocks execution; export/flush failure is surfaced without exposing credentials | Langfuse v4 SDK and LangChain callback |
| Faithfulness Evaluator (`observability/evaluation.py`, `graph/nodes.py::BVFNodes.evaluating_faithfulness`) | `make_faithfulness_evaluator` scores a whole passing report's RAGAS Faithfulness (`ragas.metrics.collections.Faithfulness`) against `source_verification.cited_source_excerpts` (the excerpt text of every source the report cites), judged by the Gemini API (`INTELLIGENCE_FAITHFULNESS_MODEL`, default `gemini-3.7-flash`) via a `google.genai.Client` patched async through `instructor.from_genai(..., use_async=True)` — bypassing `ragas.llms.llm_factory` directly, whose Google path always patches synchronously (`instructor.from_genai(client)`, no `use_async`) and breaks `Faithfulness.ascore()` with `TypeError: Cannot use agenerate() with a synchronous client` (verified against the real API) | Its own graph node (`Node.EVALUATE_FAITHFULNESS`), reached only via the `VERIFY` node's PASSED conditional edge — never as part of a build/verify/fix attempt, so a draft that fails and gets rewritten is never scored, only the one report that actually left the BVF loop (`VERIFY --PASSED--> EVALUATE_FAITHFULNESS --> APPROVE\|END`, see `build.py`); a judge-LLM failure is caught and recorded (`faithfulness_error`), never raised into the graph; score/error persist to `verification/faithfulness.<verification_id>.json` and render in the UI next to the report; disabled (`None`) unless `GOOGLE_API_KEY` is set — independent of which provider the reasoning/standard tiers actually resolve to at call time (though it happens to default to the same one) | `ragas`, `instructor[google-genai]` (needs the `jsonref` package), `google-genai` |

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
│   ├── supervisor.py       # ResearchSupervisor: delegation + ResearchComplete gate
│   ├── researcher.py       # ResearchAgent sub-agent (LLM ↔ tavily_search loop + compression)
│   ├── worker.py           # persist a sub-agent outcome to VFS + vector index
│   ├── report_writer.py    # BVF Build/Fix drafts + reflection (lessons, strategy)
│   ├── acceptance_verifier.py  # LLM goal-achievement check, run only after Verify passes
│   ├── synthesis.py        # report version persistence (draft.vN / previous / latest)
│   └── verification.py
├── tools/
│   ├── write_todos.py
│   ├── think_tool.py
│   ├── mcp.py
│   └── url_tracking.py     # normalize_url + run-level UrlRegistry (capture each page once)
├── storage/
│   ├── vfs.py
│   ├── checkpoints.py
│   ├── artifacts.py
│   ├── vector_index.py         # VectorIndex Protocol + LocalVectorIndex (lexical, offline)
│   └── chroma_vector_index.py  # ChromaVectorIndex: Gemini embeddings (default), local Chroma persistence
├── models/
│   ├── router.py       # ModelRouter: primary/fallback, tracing, structured-output parsing
│   ├── role_router.py  # RoleModelRouter: dispatches scope/planner/verify vs. the rest
│   └── factory.py      # ModelFactory: init_chat_model wiring, create_role_router
├── guardrails.py
├── publication.py
├── source_verification.py  # BVF "Verify": programmatic report-vs-sources checks + CLI
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
  and `ResearchNodes(vfs, index, researcher, acceptance_verifier=None, max_concurrent_researchers=3)`,
  which owns a `ResearchSupervisor` and a `BVFNodes` (holding the optional `AcceptanceVerifier`).
  `build_graph(planner_agent=...)` defaults to a `PlannerAgent` sharing the scope agent's model
  router (`scope_agent.router`) — in the live app this is the shared `RoleModelRouter`, so each
  agent's calls still land on the correct reasoning/standard profile by role name.
- `graph/routing.py` holds named routers for every conditional edge: `route_after_clarify`
  (END when clarification is needed, else `write_research_brief`) and `route_after_verify`
  (`approval` only when `verification.passed`; a `None`/missing result ends the run).
- `graph/instrumentation.py` provides `node_wrapper`, applied uniformly to every node for lifecycle
  events, checkpoint load/save, and Langfuse spans.
- `graph/build.py` contains only wiring. `build_graph` supports three topologies selected by
  injected collaborators:
  - Full (`scope_agent` + `research_nodes`): `clarify_with_user` → (END | `write_research_brief`) →
    `planning` → `researching` → `research_complete` → `synthesizing` → `verifying`, then the
    BVF loop: `verifying` → (`approval` | END) when every check passes; → `fixing` →
    (`synthesizing` for DRAFT_DEFECT | `researching_gaps` → `synthesizing` for EVIDENCE_GAP) while
    budget remains and progress is made; → `needs_review` → END otherwise. BVF nodes live in
    `BVFNodes` (`research_nodes.bvf`); routers are `route_after_verify` and `route_after_fix`.
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
