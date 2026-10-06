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

**Langfuse Trace Hierarchy**: A research run has one trace whose ID is derived from `run_id`
(`create_trace_id(seed=...)`), with `session_id` = LangGraph `thread_id`. Each graph invocation
(chat turn, interrupt resume, recovery action, continuation) is one execution observation,
named after the turn and the agents that ran in it (`start: Scope Agent → Planner ⏸ plan_approval`), closed when the invocation returns, so nothing stays open across human pauses or
restarts. `RunTracer.execution` creates a Langfuse `CallbackHandler` per invocation and
`Execution.graph_config` passes it in the LangGraph config: node runs (renamed
`<agent> · <node>` by `node_wrapper`), `research_agent · <todo_id>` sub-agent runs and model generations nest under the execution, and model calls inherit the
callback instead of passing their own. Non-LangChain operations (MCP, persistence, Chroma,
RAGAS) use `traced()`. Langfuse problems disable tracing with a reason and never stop research;
offline
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
4. Planning runs the Planner Agent: a structured LLM call returns the fewest standalone research
   questions (one for a simple question or comparison), the evidence each needs and the expected
   report shape. The system persists `planned` todos via `write_todos` and the hash-bound
   `planning/plan.v<n>.json`, then `await_plan_approval` interrupts; retrieval starts only after
   the exact plan is approved (FR-040). The Approve / Cancel buttons resume with
   `{"approved": true|false}`; a typed reply is always a change request.
5. The Research Supervisor is an LLM (role `supervisor`, FR-047). It receives the pending tasks
   and decides with `think_tool`, `ConductResearch(todo_id, research_topic)`, `ResearchComplete`
   and, for an original approved plan, `ProposeReplan`. The `ConductResearch` calls of one
   response run in parallel, up to `INTELLIGENCE_MAX_CONCURRENT_RESEARCHERS` (default 3). Each
   result comes back as a tool reply (status, sufficiency, gaps, findings); a second call for a
   task researches only its named gap as a new attempt whose findings are appended. Code refuses
   unknown or duplicate tasks, a third round, and `ResearchComplete` while a task was never
   researched, and researches any never-delegated task after `max_turns`.
   Each sub-agent gets a fresh Tavily MCP client and runs the loop `llm_call` → (`tool_node` →
   `llm_call`)* → `compress_research` over `tavily_search` and `think_tool`: the first decision is
   a search, the LLM reflects after each result, narrows as needed, and stops with a final note,
   at the per-task five-call hard limit, or once it holds `INTELLIGENCE_MAX_SOURCES_PER_TASK`
   distinct pages (default 3, max 10, counted across all of its searches — each search asks
   Tavily only for the slots left). Tavily returns each page's full text (`include_raw_content`);
   the tool step cleans it, splits it into the index chunks and keeps the three chunks closest to
   the research question and the search query, verbatim and in page order (`PassageSelector`,
   FR-048; no LLM call per page). Only those passages reach the researcher's context.
   Compression returns cited `ResearchFindings` from the passages, never the reflections.
   As each sub-agent returns, the Supervisor writes raw pages (`raw/<n>.md`), raw captures, notes
   (with findings), excerpts, evidence bundles, and the pages' chunks to the vector index
   (FR-049). URLs are tracked by normalized identity: a sub-agent hides pages it already retrieved
   from later search results and the run-level `research/url_registry.json` makes later tasks, gap
   research, and resumed runs reference the first capture instead of re-capturing and re-indexing.
   A `ProposeReplan` routes `researching` → `replan` → `await_plan_approval`.
6. `research_complete` derives a progress assessment with explicit blockers from the stored
   bundles. This ends Research only.
7. **Build**: the Report Writer receives the original question, required answers, output
   requirements, report shape, the `[S<n>]` source catalog and known gaps, and writes an
   answer-first draft (FR-044). A question's evidence is read directly when its excerpts fit
   12,000 characters; larger corpora are narrowed to `INTELLIGENCE_EVIDENCE_TOP_K` with Chroma.
8. **Verify**: three gates on the exact draft — coverage (`agents/report_judge.py`, id-tagged
   criteria), citation resolution (`source_verification.citation_errors`), RAGAS Faithfulness
   > 0.90 — with PASS, FAIL and ERROR outcomes (FR-039, FR-041). PASS leaves the loop at once.
9. **Fix**: `fixing` diagnoses the smallest correction (`repair.py`, no model call) and routes to
   a targeted rewrite (`vfs`, `chroma`, `presentation`), `prepare_repair_research` → research of
   only the affected tasks, or `replan` → `await_plan_approval`. The writer gets every issue, the
   answers that already passed and the previous draft. Repeating the same failure twice or
   reaching `INTELLIGENCE_RECOVERY_ATTEMPTS` (default 3) ends in `needs_review`; ERROR never
   rewrites (FR-042).
10. A passing result enters `Awaiting Approval` and LangGraph interrupts. The user approves or
    rejects the exact artifact hash. Approval resumes the graph; rejection records the decision and
    leaves permanent output unchanged.
11. Publication verifies the artifact hash and verification status, writes the permanent artifact
    atomically, records a Publication Record, and transitions to Published.

## Component Responsibilities and Boundaries

| Component | Responsibility | Contract / failure behavior | Integration points |
|---|---|---|---|
| Streamlit UI (`app.py` entry, package `ui/`, `.streamlit/config.toml`) | `ui/main.py` composes the page; `ui/state.py` owns session-state keys, defaults and every change (`on_click` callbacks for New thread and quick replies, which queue a reply answered like a typed one; Request changes is an `st.form`); `ui/resources.py` builds per-session dependencies (tracer, models, graph, VFS, recovery service); `ui/turn.py` runs one chat turn (lifecycle log, `graph.stream(stream_mode="values")`, live supervisor decisions via `decision_listener`); `ui/components/` render header + stepper, chat (plan and report cards, banner, notices, input), the tabbed panel (`LivePanel` placeholders redrawn every step) and recovery | `ui/run_status.py` (`RunView`) and `ui/evidence.py` derive everything from the snapshot without Streamlit; components never write authoritative artifacts; agent cards stay hidden until their agent leaves `pending`; HTML fragments in `ui/html.py` escape all text, styles in `ui/theme.css` | Graph runner, VFS read API (evidence bundles, supervisor log) |
| Input Normalizer + Guardrail (`guardrails.py`) | Fold homoglyphs/compatibility forms and strip invisible characters (`normalize_text`); validate shape/size and reject prompt injection or disallowed offensive/violent requests via regex | Returns typed `GuardrailDecision`; rejection has no state mutation | Scope entry edge, audit event |
| Scope Agent | `check_content_safety` runs a pre-model LLM guardrail (`guardrail` role, standard tier) that blocks vulgar/prohibited messages before any Scope call; `clarify_with_user` evaluates actual message history; `write_research_brief` creates a dedicated validated `ResearchBrief` after a clear decision | Blocked content raises `GuardrailViolation` with zero Scope-model calls; otherwise one-question clarification or complete brief; model/validation/persistence failure cannot become clear | Model router (`guardrail` role, standard tier; `scope` role, reasoning tier), LangGraph thread state, checkpoint, brief VFS artifact |
| Planner Agent + `write_todos` | `PlannerAgent.define_plan` has the planner LLM call the `write_todos` tool (`agents/tools.py::WRITE_TODOS_TOOL`) with the research questions; `PlannerAgent.write_todos` converts the call's questions to `planned` todos and persists them | `write_todos` arguments validated as `TodoPlan` (questions, 1–5, non-blank, unique); an invalid call gets its error back as a `ToolMessage` and is retried; repeated invalid output or failed write blocks Research | Model router (`planner` role, reasoning tier), VFS, graph state |
| Report Judge (`agents/report_judge.py`) | Sole Verify gate: one LLM-as-judge call (no search/scrape tools) re-reads the report against the already-gathered evidence and decides factual accuracy (no inaccuracies/hallucinations/inconsistencies introduced by the Writer — the fact-check pass) plus the operator-configured guideline checklist (`Settings.report_guidelines`, e.g. Markdown format, sourced sub-sections, English) — there is no separate deterministic pre-check | `ReportJudgeVerdict` (met/unmet + justification per criterion); any unmet criterion fails the `VerificationResult` and goes to the reflect-and-rewrite fix | Model router (`verify` role, reasoning tier), evidence catalog, `Settings.report_guidelines` |
| Research Supervisor (`agents/supervisor.py`) | LLM tool loop over `think_tool`, `ConductResearch`, `ResearchComplete`, `ProposeReplan`; parallel batches (bounded); follow-ups append findings; persist each outcome | Refuses unknown/duplicate tasks, extra rounds and premature ResearchComplete; never skips an approved task; a failing sub-agent becomes a FAILED outcome, never cancels siblings; a failed follow-up keeps earlier evidence | Model router (`supervisor` role), `ResearchAgent`, VFS, vector store |
| Research Sub-agent (`agents/researcher.py`) | LangGraph loop: LLM decision → `tavily_search` / `think_tool` execution → routing → research compression | First decision is a search; reflections never spend the search budget nor reach compression; fresh MCP client and hard per-task budget (≤5); MCP failures become tool/task outcomes with redacted errors; `ResearchFindings` validated | Model router (`researcher`, `research_compressor` roles, tool calling), Tavily MCP, `PassageSelector` |
| Passage selector (`agents/passage_selector.py`) | Clean each fetched page, rank its index chunks against the research question and search query, keep the best three verbatim | No LLM call; one embedding call per search; embedding failure falls back to word overlap; short pages kept whole, unusable pages keep the snippet; vectors reused by the index (`storage/embedding_cache.py`) | Embeddings (`CachingEmbeddings`), `tools/page_content.py`, `storage/evidence_index.py` |
| Research persistence (`agents/worker.py`) | Turn a `ResearchOutcome` into raw pages, raw captures, notes, excerpts, evidence bundle, and page chunks; idempotent resume guard; capture each URL once per run | COMPLETED with sources + findings (gaps recorded); FAILED writes `outcome.json`; already-captured URLs reference the first capture (`reused_from_task`) | VFS, vector store, `UrlRegistry`, `storage/evidence_index.py` |
| Evidence index (`storage/evidence_index.py`) | Build stable chunk records from stored pages; mark which index holds a run; rebuild a run for a new index | Chunk ids `<run>:<source_id>:<n>` (upserts, no duplicates); rebuild happens once per index identity | VFS, vector store |
| VFS Artifact Store | Atomic run-scoped files and immutable versions | Reject path escape/cross-run writes; partial writes are invisible | All nodes, checkpoint metadata |
| Vector Index | Chunk/index raw source captures and retrieve synthesis context | Index failures block ResearchComplete; retrieval records source IDs | Worker, synthesis, verifier |
| Report Writer (`agents/report_writer.py`) | Build/Fix drafts from top-K retrieved evidence (`select_relevant_evidence`, per question) with `[S<n>]` citations; BVF reflection producing lessons and strategy | `DraftAssessment` / `Reflection` validated; fix attempts must receive failures, lessons, strategy, previous draft; retrieval scoped per question, never global | Model router (`synthesizer`, `fixer` roles), VFS, vector index |
| Vector Index (`storage/vector_index.py`, `storage/chroma_vector_index.py`) | `LocalVectorIndex` (lexical, offline/tests) and `ChromaVectorIndex` (Gemini embeddings by default, OpenAI selectable via `INTELLIGENCE_EMBEDDING_PROVIDER`, local Chroma persistence, live) behind one `VectorIndex` interface | `retrieve(query, limit, run_id, todo_id)`; Verify never reads through this — it always loads the run's un-narrowed catalog from VFS independently (every stored evidence bundle with the passages selected from its pages; raw pages stay in the VFS and the index) | Gemini/OpenAI embeddings API, local Chroma directory |
| Evidence catalog (`source_verification.py`) | Loads a run's evidence bundles into a catalog with stable `[S<n>]` source ids and the sub-agents' unresolved gaps; cited-excerpt lookup for RAGAS; failure fingerprint for BVF stall detection (the deterministic checker was removed) | Catalog built only from stored bundles | Evidence bundles in VFS, `verification/catalog_state.json` |
| Model Router (`models/router.py`, `models/factory.py`, `models/redaction.py`) | `ModelFactory.create_routers` builds two `ModelRouter` tiers and each agent is constructed with the tier it needs: reasoning (Scope, Planner, Report Judge) or standard (Researcher, Report Writer, Feedback classifier, and the scope guardrail via `ScopeAgent(guardrail_router=...)`) — the "Reasoning Sandwich": a high-effort model plans and verifies around standard-tier implementation work. Live app default: Gemini for both tiers (`INTELLIGENCE_REASONING_MODEL/PROVIDER` = `gemini-3.8-flash`/`google_genai` at `reasoning_effort=high`, the ceiling `ChatGoogleGenerativeAI` accepts — neither Gemini nor OpenAI's SDK has an "xhigh" literal, that word is only this tier's name; `INTELLIGENCE_STANDARD_MODEL/PROVIDER` = `gemini-3.5-flash-lite`/`google_genai` at `reasoning_effort=low`), each with a Groq fallback (`openai/gpt-oss-120b` for reasoning, `openai/gpt-oss-20b` for standard), built once via `app._get_model_routers`; the fallback is LangChain `with_fallbacks` | Structured-output retries bounded per profile; fallback model run is visible in the Langfuse `model.<role>` span; final failure is visible | All model-using nodes, Langfuse |
| Checkpointer | Persist typed graph state and resume points | Atomic checkpoint; invalid latest checkpoint falls back or NEEDS_REVIEW | LangGraph, VFS metadata |
| Approval/Publisher (`graph/nodes.py::ApprovalNodes`, `publication.py`, `agents/feedback_router.py`) | Human review flow, wired only when `with_approval=True`: `evaluating_quality` -> `Node.APPROVE` (LangGraph `interrupt()`/`Command(resume={"reply": ...})`; an explicit `{"approved": True}` goes straight to publishing). Every approval binds to the hash of the verified report shown. A reply goes to `Node.CLASSIFY_FEEDBACK`, a separate LLM node (a failed classification never re-runs the interrupt), where `FeedbackRouter` returns `"approve"` -> `Node.PUBLISH` (atomic write to `{output_dir}/{run_id}.md`), `"revise"` -> the request joins `reviewer_requests`, counters restart, and `Node.SYNTHESIZE` rewrites the reviewed report from the same evidence (the requests override conflicting layout guidelines for the writer and the judge, never the evidence rules) before Verify runs again, `"extend"` (FR-045) -> the request joins `reviewer_requests` and `Node.LOCATE_EXTENSION` (`BVFNodes.locate_extension_evidence`) retrieves up to 8 stored passages for the router's `topic` from the run's vector index; `ExtensionJudge` (`agents/extension_judge.py`, role `extension_judge`, standard tier) returns `ExtensionEvidenceVerdict`; enough evidence adds a COMPLETED task whose bundle holds only the relevant passages and goes to `Node.SYNTHESIZE`, otherwise a PLANNED task with the missing points goes to `Node.RESEARCH`; both add an auto-approved plan revision of kind `extension` and record `extension/<n>.json`, or `"scope"` -> the feedback becomes the next user turn and the run re-enters `Node.CLARIFY` -> `write_research_brief` -> `Node.PLAN` -> research, with reviewer requests cleared and counters restarted; unclear, empty or unclassified replies are `"revise"` | Reject/stale-hash/unverified `publish()` writes nothing permanent; routing is `route_after_approval` (over `approval.approved`) then `route_after_feedback` (over `feedback_route`) | LangGraph interrupt/resume, VFS, output path, Streamlit chat |
| Tracing/Evaluation | Create root execution traces grouped by thread, nested node spans, model generations (including fallback model runs), and validation events | Missing live tracing config blocks execution; export/flush failure is surfaced without exposing credentials | Langfuse v4 SDK and LangChain callback |
| Faithfulness Evaluator (`observability/evaluation.py`, `graph/nodes.py::BVFNodes.evaluating_faithfulness`) | `make_faithfulness_evaluator` scores a whole passing report's RAGAS Faithfulness (`ragas.metrics.collections.Faithfulness`) against `source_verification.cited_source_excerpts` (the excerpt text of every source the report cites), judged by the Gemini API (`INTELLIGENCE_FAITHFULNESS_MODEL`, default `gemini-3.7-flash`) via a `google.genai.Client` patched async through `instructor.from_genai(..., use_async=True)` — bypassing `ragas.llms.llm_factory` directly, whose Google path always patches synchronously (`instructor.from_genai(client)`, no `use_async`) and breaks `Faithfulness.ascore()` with `TypeError: Cannot use agenerate() with a synchronous client` (verified against the real API) | Its own graph node (`Node.EVALUATE_FAITHFULNESS`), reached only via the `VERIFY` node's PASSED conditional edge — never as part of a build/verify/fix attempt, so a draft that fails and gets rewritten is never scored, only the one report that actually left the BVF loop (`VERIFY --PASSED--> EVALUATE_FAITHFULNESS --> APPROVE\|END`, see `build.py`); a judge-LLM failure is caught and recorded (`faithfulness_error`), never raised into the graph; score/error persist to `verification/faithfulness.<verification_id>.json` and render in the UI next to the report; judged by `INTELLIGENCE_FAITHFULNESS_MODEL`/`_PROVIDER` (default OpenAI `gpt-4.1-mini` via an `AsyncOpenAI` client; `google_genai` also supported), disabled (`None`) unless that provider's key (`OPENAI_API_KEY`/`GOOGLE_API_KEY`) is set — independent of the reasoning/standard tier providers | `ragas`, `instructor[google-genai]` (needs the `jsonref` package), `google-genai` |
| Context Precision and Response Relevancy (`observability/evaluation.py`, `graph/nodes.py::BVFNodes.evaluating_quality`) | `synthesizing` persists the per-question ranked source ids the writer saw (`writer_evidence`, `synthesis/retrieval.v<n>.json`). `make_context_precision_evaluator` scores `ContextPrecisionWithoutReference` per research question on those excerpts in rank order against the report body (mean over questions, per-question details kept); `make_response_relevancy_evaluator` scores `AnswerRelevancy` of the report body against the brief, with the vector index's LangChain embeddings adapted to RAGAS's `BaseRagasEmbedding`. Same judge LLM as Faithfulness | One `evaluating_quality` node (renamed from `evaluating_faithfulness`) runs every metric in `INTELLIGENCE_EVAL_METRICS` (default all); each failure is recorded as `<metric>_error` without affecting the others; all scores persist to `verification/quality.<verification_id>.json`, render in the UI and become Langfuse trace scores | `ragas` collections metrics, `langchain_google_genai`/`langchain_openai` embeddings |

## Project Structure

```text
src/intelligence_collector/
├── ui/                 # Streamlit UI (entry: ../../app.py -> ui/main.py)
│   ├── main.py         # page composition; answers a reply with one graph turn
│   ├── state.py        # session-state keys, defaults, on_click callbacks, reply queue
│   ├── resources.py    # per-session tracer, models, graph, VFS, recovery service
│   ├── turn.py         # one chat turn: streaming, lifecycle log, live supervisor decisions
│   ├── run_status.py   # RunView: who acts, input placeholder, agent status (pure)
│   ├── evidence.py     # stored research read for display (pure)
│   ├── html.py + theme.css  # escaped HTML fragments and their styles
│   └── components/     # header, chat, panel (tabs, LivePanel), recovery
├── runtime.py          # composition root: builds models, agents, stores and the graph (no Streamlit)
├── config.py           # Settings, RoleModelSettings (reasoning + standard tiers), LangfuseSettings
├── contracts.py        # Pydantic contracts (stages, LLM outputs, verification, approval, publication)
├── guardrails.py       # deterministic request checks + content-safety verdict contract
├── publication.py      # approval binding + atomic publish
├── source_verification.py  # evidence catalog ([S<n>] ids, gaps), cited excerpts, failure fingerprint
├── graph/
│   ├── state.py            # GraphState (no reducers: nodes return only the keys they change)
│   ├── build.py            # wiring only: nodes + edges per topology
│   ├── nodes/              # node logic by phase: common, interrupts, scope, research, report,
│   │                       #   review, recovery (re-exported from __init__); common holds
│   │                       #   invalidate_draft_authority()
│   ├── routing.py          # named conditional-edge routers
│   └── instrumentation.py  # node_wrapper: lifecycle notifications + tracing; merges state +
│                           #   update for registry bookkeeping and persists checkpoint migrations
├── agents/
│   ├── prompts.py          # every role's ChatPromptTemplate
│   ├── scope.py            # ScopeAgent: guardrail, clarify, research brief
│   ├── planner.py          # PlannerAgent + create_todos
│   ├── supervisor.py       # ResearchSupervisor: LLM tool loop, parallel delegation, follow-ups
│   ├── researcher.py       # ResearchAgent sub-agent (LLM <-> tavily_search/think_tool + compression)
│   ├── passage_selector.py # PassageSelector: page -> most relevant verbatim chunks
│   ├── tools.py            # tool schemas: tavily_search, think_tool, ConductResearch, write_todos, ...
│   ├── worker.py           # persist a sub-agent outcome to VFS + vector index
│   ├── report_writer.py    # BVF Build/Fix drafts, reviewer revisions, reflection
│   ├── report_judge.py     # BVF Verify: LLM-as-judge
│   ├── feedback_router.py  # classifies the reviewer's chat reply (revise / extend / scope)
│   └── extension_judge.py  # judges whether stored passages cover a section the reviewer added
├── tools/
│   ├── write_todos.py
│   ├── mcp.py              # MCPResearchPort, ControlledMCP (offline), TavilyMCP (+ raw content)
│   ├── page_content.py     # clean/cap raw pages, chunk_page, stable source_id_for
│   └── url_tracking.py     # normalize_url + run-level UrlRegistry (capture each page once)
├── storage/
│   ├── vfs.py
│   ├── reports.py              # report version persistence (draft.vN / previous / latest)
│   ├── evidence_index.py       # page chunks from the VFS; rebuild a run for a new index
│   ├── vector_index.py         # VectorIndex Protocol + LocalVectorIndex (lexical, offline)
│   └── chroma_vector_index.py  # ChromaVectorIndex: Gemini embeddings (default), local Chroma persistence
├── ports.py            # Protocols: ModelRouterPort, ArtifactStore, VectorIndex, MCPResearchPort
├── models/
│   ├── router.py       # ModelRouter: primary/fallback, structured-output parsing (traced via callbacks)
│   ├── factory.py      # ModelFactory: init_chat_model wiring, create_routers (reasoning + standard)
│   └── redaction.py    # redact_text / redact_value / redact_error: model, compression, MCP,
│                       #   embedding-log and trace errors
└── observability/
    ├── tracing.py      # Langfuse: one trace per chat thread, node spans, scores
    └── evaluation.py   # RAGAS judges: Faithfulness, Context Precision, Response Relevancy

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

- `graph/nodes/` (a package, one module per phase) owns the node logic: `common.py` (the `Node`
  name constants, the single source for wiring, `app.py` activity labels and trace observation
  names, plus workflow limits), `interrupts.py` (every human-in-the-loop pause), `scope.py`
  (`ScopeNodes`, `PlannerNodes`), `research.py` (`ResearchNodes`), `report.py` (`BVFNodes`),
  `review.py` (`ApprovalNodes`) and `recovery.py` (`RecoveryNodes`). Its
  `__init__` re-exports every public name, so `from intelligence_collector.graph.nodes import
  ...` is unchanged. `ResearchNodes` owns a `ResearchSupervisor` and a `BVFNodes`.
  `build_graph(planner_agent=...)` defaults to a `PlannerAgent` sharing the scope agent's model
  router (`scope_agent.router`) — in the live app this is the reasoning-tier router, while `ResearchAgent`
  and `ReportWriter` are constructed with the standard-tier router.
- `graph/routing.py` holds named routers for every conditional edge: `route_after_clarify`
  (END when clarification is needed, else `write_research_brief`) and `route_after_verify`
  (`approval` only when `verification.passed`; a `None`/missing result ends the run).
- `graph/instrumentation.py` provides `node_wrapper`, applied uniformly to every node for lifecycle
  events, checkpoint load/save, and Langfuse spans.
- `graph/build.py` contains only the production wiring. `build_graph` requires `scope_agent`
  and supports two topologies:
  - Full (`scope_agent` + `research_nodes`): `clarify_with_user` → (END | `write_research_brief`) →
    `planning` → `researching` → `research_complete` → `synthesizing` → `verifying`, then the
    BVF loop: `verifying` → `evaluating_quality` → `approval` when every check
    passes; → `fixing` → `synthesizing` (rewrite) while budget remains and progress is made;
    → `needs_review` → END otherwise. BVF nodes live in `BVFNodes` (`research_nodes.bvf`); the
    router is `route_after_verify`.
  - Scope only (`scope_agent`, no `research_nodes`, e.g. Tavily not configured): … →
    `planning` → END.
  Tests that inject a brief and todos enter the production graph at research through
  `tests/fixtures/graphs.py::research_graph` (the input is recorded as an approved plan);
  `approval_graph` there is a bare publication interrupt for checkpoint/recovery tests.

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
