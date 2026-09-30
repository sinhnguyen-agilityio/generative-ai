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

**LLM Decision Contract**: Semantic roles use `ModelRouter.invoke_messages()` with explicit
prompt templates and Pydantic output schemas. The router requires a configured primary model
(there is no silent fake; tests inject fakes explicitly). Router traces role,
primary model name, validation errors, and validated decision metadata without secrets
(`models/redaction.py`); a fallback model's run appears in the Langfuse callback trace.

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
   second structured call with its own prompt, validates a dedicated single-question
   `ResearchBrief`,
   writes `scope/brief_summary.v1.json`, and enters Planning without a brief approval interrupt.
4. Planning runs the Planner Agent: a structured LLM call decomposes the brief's
   research question into 1–5 standalone, non-overlapping research questions (`TodoPlan.questions`).
   The planner no longer defines acceptance criteria (removed 2026-09-30). The system turns each
   question into a `planned` todo and persists the list via `write_todos`. A non-empty, valid,
   durably stored todo list is required before any Research/retrieval edge.
5. The Research Supervisor delegates each planned todo to its own research sub-agent, running up to
   `INTELLIGENCE_MAX_CONCURRENT_RESEARCHERS` (default 3) in parallel. Each sub-agent gets a fresh
   Tavily MCP client and runs the loop `llm_call` → (`tool_node` → `llm_call`)* →
   `compress_research`: the LLM starts broad, assesses gaps after each
   `tavily_search` result, narrows as needed, and stops with a final note, at the per-task
   five-call hard limit, or once it holds `INTELLIGENCE_MAX_SOURCES_PER_TASK` distinct pages
   (default 5, max 10, counted across all of its searches — each search asks Tavily only for the
   slots left); compression returns cited `ResearchFindings`.
   After all sub-agents return, the Supervisor writes raw source captures, notes (with findings),
   excerpts, evidence bundles, and vector-index records to the VFS in plan order. URLs are tracked
   by normalized identity: a sub-agent hides pages it already retrieved from later search results
   and the run-level `research/url_registry.json` makes later tasks, gap
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
   unfiltered. The Report Writer (structured LLM call) then drafts the report
   from that narrowed evidence, citing the (unchanged, still-complete) per-run source catalog as
   `[S<n>]`, and persists `reports/draft.v<n>.md`, `reports/previous.md`, and `reports/latest.md`.
8. **Verify**: `ReportJudge` (`agents/report_judge.py`, role `verify`) is the sole Verify gate: one
   LLM call with no search/scrape tools re-reads the report against the already-gathered evidence
   and checks the fact-check criterion plus the configured report guidelines. Each criterion gets
   met/unmet and a justification; failures are not classified. Only a report passing every
   criterion leaves the loop.
9. **Fix** (self-improving, rewrite only): each failure increments the visible attempt counter. A
   reflection call diagnoses the root cause, adds reusable lessons to an accumulating lesson
   memory, and plans a strategy; the next Build rewrites the report from the same evidence,
   receiving the failed checks, all lessons, the strategy, and the previous draft. No new research
   happens inside the loop: an unsupported claim is removed, narrowed, or disclosed under
   `## Limitations`. A failure fingerprint detects no progress: the first repeat tells the
   reflection to change approach; two consecutive repeats, or reaching
   `INTELLIGENCE_RECOVERY_ATTEMPTS` (default 3), end in `needs_review` (`NEEDS_REVIEW`).
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
| Planner Agent + `write_todos` | `PlannerAgent.define_plan` turns the `ResearchBrief` into research questions; `PlannerAgent.write_todos` converts the questions to `planned` todos and persists them | `TodoPlan` (questions, 1–5, non-blank, unique); invalid output or failed write blocks Research | Model router (`planner` role, reasoning tier), VFS, graph state |
| Report Judge (`agents/report_judge.py`) | Sole Verify gate: one LLM-as-judge call (no search/scrape tools) re-reads the report against the already-gathered evidence and decides factual accuracy (no inaccuracies/hallucinations/inconsistencies introduced by the Writer — the fact-check pass) plus the operator-configured guideline checklist (`Settings.report_guidelines`, e.g. Markdown format, sourced sub-sections, English) — there is no separate deterministic pre-check | `ReportJudgeVerdict` (met/unmet + justification per criterion); any unmet criterion fails the `VerificationResult` and goes to the reflect-and-rewrite fix | Model router (`verify` role, reasoning tier), evidence catalog, `Settings.report_guidelines` |
| Research Supervisor (`agents/supervisor.py`) | Delegate one sub-agent per todo (bounded thread pool), persist outcomes in plan order, use `think_tool`, assess readiness | `ProgressAssessment` with blockers; a failing sub-agent becomes a FAILED task outcome, never cancels siblings; cannot ResearchComplete until all todos complete and artifacts indexed | `ResearchAgent`, VFS, vector store |
| Research Sub-agent (`agents/researcher.py`) | LangGraph loop: LLM decision → `tavily_search` tool execution → routing → research compression | Fresh MCP client and hard per-task budget (≤5); MCP failures become tool/task outcomes with redacted errors; `ResearchFindings` validated | Model router (`researcher`, `research_compressor` roles, tool calling), Tavily MCP |
| Research persistence (`agents/worker.py`) | Turn a `ResearchOutcome` into raw captures, notes, excerpts, evidence bundle, and index chunks; idempotent resume guard; capture each URL once per run | COMPLETED with sources + findings (gaps recorded); FAILED writes `outcome.json`; already-captured URLs reference the first capture (`reused_from_task`) | VFS, vector store, `UrlRegistry` |
| VFS Artifact Store | Atomic run-scoped files and immutable versions | Reject path escape/cross-run writes; partial writes are invisible | All nodes, checkpoint metadata |
| Vector Index | Chunk/index raw source captures and retrieve synthesis context | Index failures block ResearchComplete; retrieval records source IDs | Worker, synthesis, verifier |
| Report Writer (`agents/report_writer.py`) | Build/Fix drafts from top-K retrieved evidence (`select_relevant_evidence`, per question) with `[S<n>]` citations; BVF reflection producing lessons and strategy | `DraftAssessment` / `Reflection` validated; fix attempts must receive failures, lessons, strategy, previous draft; retrieval scoped per question, never global | Model router (`synthesizer`, `fixer` roles), VFS, vector index |
| Vector Index (`storage/vector_index.py`, `storage/chroma_vector_index.py`) | `LocalVectorIndex` (lexical, offline/tests) and `ChromaVectorIndex` (Gemini embeddings by default, OpenAI selectable via `INTELLIGENCE_EMBEDDING_PROVIDER`, local Chroma persistence, live) behind one `VectorIndex` interface | `retrieve(query, limit, run_id, todo_id)`; Verify never reads through this — it always loads the full evidence catalog from VFS independently | Gemini/OpenAI embeddings API, local Chroma directory |
| Evidence catalog (`source_verification.py`) | Loads a run's evidence bundles into a catalog with stable `[S<n>]` source ids and the sub-agents' unresolved gaps; cited-excerpt lookup for RAGAS; failure fingerprint for BVF stall detection (the deterministic checker was removed) | Catalog built only from stored bundles | Evidence bundles in VFS, `verification/catalog_state.json` |
| Model Router (`models/router.py`, `models/factory.py`, `models/redaction.py`) | `ModelFactory.create_routers` builds two `ModelRouter` tiers and each agent is constructed with the tier it needs: reasoning (Scope, Planner, Report Judge) or standard (Researcher, Report Writer, Feedback classifier, and the scope guardrail via `ScopeAgent(guardrail_router=...)`) — the "Reasoning Sandwich": a high-effort model plans and verifies around standard-tier implementation work. Live app default: Gemini for both tiers (`INTELLIGENCE_REASONING_MODEL/PROVIDER` = `gemini-3.8-flash`/`google_genai` at `reasoning_effort=high`, the ceiling `ChatGoogleGenerativeAI` accepts — neither Gemini nor OpenAI's SDK has an "xhigh" literal, that word is only this tier's name; `INTELLIGENCE_STANDARD_MODEL/PROVIDER` = `gemini-3.5-flash-lite`/`google_genai` at `reasoning_effort=low`), each with a Groq fallback (`openai/gpt-oss-120b` for reasoning, `openai/gpt-oss-20b` for standard), built once via `app._get_model_routers`; the fallback is LangChain `with_fallbacks` | Structured-output retries bounded per profile; fallback model run is visible in the Langfuse `model.<role>` span; final failure is visible | All model-using nodes, Langfuse |
| Checkpointer | Persist typed graph state and resume points | Atomic checkpoint; invalid latest checkpoint falls back or NEEDS_REVIEW | LangGraph, VFS metadata |
| Approval/Publisher (`graph/nodes.py::ApprovalNodes`, `publication.py`, `agents/feedback_router.py`) | Human review flow, wired only when `with_approval=True`: `evaluating_faithfulness` -> `Node.APPROVE` (LangGraph `interrupt()`/`Command(resume={"reply": ...})`; an explicit `{"approved": True}` goes straight to publishing). Every approval binds to the hash of the verified report shown. A reply goes to `Node.CLASSIFY_FEEDBACK`, a separate LLM node (a failed classification never re-runs the interrupt), where `FeedbackRouter` returns `"approve"` -> `Node.PUBLISH` (atomic write to `{output_dir}/{run_id}.md`), `"revise"` -> the request joins `reviewer_requests`, counters restart, and `Node.SYNTHESIZE` rewrites the reviewed report from the same evidence (the requests override conflicting layout guidelines for the writer and the judge, never the evidence rules) before Verify runs again, or `"scope"` -> the feedback becomes the next user turn and the run re-enters `Node.CLARIFY` -> `write_research_brief` -> `Node.PLAN` -> research, with reviewer requests cleared and counters restarted; unclear, empty or unclassified replies are `"revise"` | Reject/stale-hash/unverified `publish()` writes nothing permanent; routing is `route_after_approval` (over `approval.approved`) then `route_after_feedback` (over `feedback_route`) | LangGraph interrupt/resume, VFS, output path, Streamlit chat |
| Tracing/Evaluation | Create root execution traces grouped by thread, nested node spans, model generations (including fallback model runs), and validation events | Missing live tracing config blocks execution; export/flush failure is surfaced without exposing credentials | Langfuse v4 SDK and LangChain callback |
| Faithfulness Evaluator (`observability/evaluation.py`, `graph/nodes.py::BVFNodes.evaluating_faithfulness`) | `make_faithfulness_evaluator` scores a whole passing report's RAGAS Faithfulness (`ragas.metrics.collections.Faithfulness`) against `source_verification.cited_source_excerpts` (the excerpt text of every source the report cites), judged by the Gemini API (`INTELLIGENCE_FAITHFULNESS_MODEL`, default `gemini-3.7-flash`) via a `google.genai.Client` patched async through `instructor.from_genai(..., use_async=True)` — bypassing `ragas.llms.llm_factory` directly, whose Google path always patches synchronously (`instructor.from_genai(client)`, no `use_async`) and breaks `Faithfulness.ascore()` with `TypeError: Cannot use agenerate() with a synchronous client` (verified against the real API) | Its own graph node (`Node.EVALUATE_FAITHFULNESS`), reached only via the `VERIFY` node's PASSED conditional edge — never as part of a build/verify/fix attempt, so a draft that fails and gets rewritten is never scored, only the one report that actually left the BVF loop (`VERIFY --PASSED--> EVALUATE_FAITHFULNESS --> APPROVE\|END`, see `build.py`); a judge-LLM failure is caught and recorded (`faithfulness_error`), never raised into the graph; score/error persist to `verification/faithfulness.<verification_id>.json` and render in the UI next to the report; judged by `INTELLIGENCE_FAITHFULNESS_MODEL`/`_PROVIDER` (default OpenAI `gpt-4.1-mini` via an `AsyncOpenAI` client; `google_genai` also supported), disabled (`None`) unless that provider's key (`OPENAI_API_KEY`/`GOOGLE_API_KEY`) is set — independent of the reasoning/standard tier providers | `ragas`, `instructor[google-genai]` (needs the `jsonref` package), `google-genai` |

## Project Structure

```text
src/intelligence_collector/
├── app.py              # Streamlit UI: chat + framed Pipeline / Activity log / Details panel
├── config.py           # Settings, RoleModelSettings (reasoning + standard tiers), LangfuseSettings
├── contracts.py        # Pydantic contracts (stages, LLM outputs, verification, approval, publication)
├── guardrails.py       # deterministic request checks + content-safety verdict contract
├── publication.py      # approval binding + atomic publish
├── source_verification.py  # evidence catalog ([S<n>] ids, gaps), cited excerpts, failure fingerprint
├── graph/
│   ├── state.py            # GraphState
│   ├── build.py            # wiring only: nodes + edges per topology
│   ├── nodes.py            # Node names, Scope/Planner/Research/BVF/Approval nodes, stubs
│   ├── routing.py          # named conditional-edge routers
│   └── instrumentation.py  # node_wrapper: lifecycle notifications + tracing
├── agents/
│   ├── prompts.py          # every role's ChatPromptTemplate
│   ├── scope.py            # ScopeAgent: guardrail, clarify, research brief
│   ├── planner.py          # PlannerAgent + create_todos
│   ├── supervisor.py       # ResearchSupervisor: parallel delegation + ResearchComplete gate
│   ├── researcher.py       # ResearchAgent sub-agent (LLM <-> tavily_search loop + compression)
│   ├── worker.py           # persist a sub-agent outcome to VFS + vector index
│   ├── report_writer.py    # BVF Build/Fix drafts, reviewer revisions, reflection
│   ├── report_judge.py     # BVF Verify: LLM-as-judge
│   ├── feedback_router.py  # classifies the reviewer's chat reply (approve / revise / scope)
│   └── synthesis.py        # report version persistence (draft.vN / previous / latest)
├── tools/
│   ├── write_todos.py
│   ├── think_tool.py
│   ├── mcp.py              # MCPResearchPort, ControlledMCP (offline), TavilyMCP
│   └── url_tracking.py     # normalize_url + run-level UrlRegistry (capture each page once)
├── storage/
│   ├── vfs.py
│   ├── vector_index.py         # VectorIndex Protocol + LocalVectorIndex (lexical, offline)
│   └── chroma_vector_index.py  # ChromaVectorIndex: Gemini embeddings (default), local Chroma persistence
├── models/
│   ├── router.py       # ModelRouter: primary/fallback, tracing, structured-output parsing
│   ├── factory.py      # ModelFactory: init_chat_model wiring, create_routers (reasoning + standard)
│   └── redaction.py
└── observability/
    ├── tracing.py      # Langfuse: one trace per chat thread, node spans, scores
    └── evaluation.py   # RAGAS Faithfulness judge

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
  router (`scope_agent.router`) — in the live app this is the reasoning-tier router, while `ResearchAgent`
  and `ReportWriter` are constructed with the standard-tier router.
- `graph/routing.py` holds named routers for every conditional edge: `route_after_clarify`
  (END when clarification is needed, else `write_research_brief`) and `route_after_verify`
  (`approval` only when `verification.passed`; a `None`/missing result ends the run).
- `graph/instrumentation.py` provides `node_wrapper`, applied uniformly to every node for lifecycle
  events, checkpoint load/save, and Langfuse spans.
- `graph/build.py` contains only wiring. `build_graph` supports three topologies selected by
  injected collaborators:
  - Full (`scope_agent` + `research_nodes`): `clarify_with_user` → (END | `write_research_brief`) →
    `planning` → `researching` → `research_complete` → `synthesizing` → `verifying`, then the
    BVF loop: `verifying` → `evaluating_faithfulness` → `approval` when every check
    passes; → `fixing` → `synthesizing` (rewrite) while budget remains and progress is made;
    → `needs_review` → END otherwise. BVF nodes live in `BVFNodes` (`research_nodes.bvf`); the
    router is `route_after_verify`.
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
