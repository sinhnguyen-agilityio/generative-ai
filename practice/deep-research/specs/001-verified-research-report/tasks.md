# Tasks: Verified Research Report

**Input**: Design documents from `/specs/001-verified-research-report/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Organization**: Tasks are ordered for an end-to-end vertical slice first, then grouped by the
five user stories. Every task includes an exact path and a completion condition with specification
references. Tests are included because the specification and request explicitly require them.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the local Python application and testable package layout.

- [X] T001 Create the package tree from `plan.md` under `src/intelligence_collector/` and `tests/`; completion means every planned module directory exists with package markers where needed (plan Project Structure).
- [X] T002 Initialize Python 3.11 project dependencies and commands in `pyproject.toml`; completion means Streamlit, direct LangGraph, Pydantic, MCP client, vector-index adapter, Langfuse, Ragas, and pytest dependencies are declared without Deep Agents SDK (plan Technical Context; FR-030).
- [X] T003 [P] Add repository configuration and safe local environment examples in `config.example.toml`, `.env.example`, and `.gitignore`; completion means secrets, VFS data, checkpoints, vector indexes, and permanent outputs are excluded from source control (constitution; FR-021).
- [X] T004 [P] Add test fixtures and deterministic fake ports in `tests/conftest.py`, `tests/fixtures/fake_models.py`, `tests/fixtures/fake_mcp.py`, and `tests/fixtures/fake_vector.py`; completion means tests can run without network or paid model calls (quickstart).
- [X] T005 [P] Add baseline lint/type/test commands in `pyproject.toml` and `tests/unit/test_imports.py`; completion means the package imports and the baseline test command runs successfully (plan Testing).

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the contracts, persistence, safety, and graph foundation required by every user story.

**Checkpoint**: No user-story implementation starts until this phase passes its contract and persistence tests.

- [X] T006 Define Pydantic domain contracts and enums in `src/intelligence_collector/contracts.py` for `ResearchGraphState`, `ScopedRequest`, `ResearchTodo`, `SourceRecord`, `ArtifactRef`, `IndexedChunk`, `ProgressAssessment`, `VerificationResult`, `ApprovalDecision`, and `PublicationRecord`; completion means required fields, stable IDs, schema versions, and enum values match `data-model.md` and contracts (FR-007, FR-016, FR-024).
- [X] T007 [P] Implement path-safe atomic VFS repository in `src/intelligence_collector/storage/vfs.py`; completion means run-scoped writes reject traversal/cross-run paths, use atomic replacement, return content hashes, and never expose partial files (FR-010, FR-010b, FR-028a).
- [X] T008 [P] Implement artifact repository and stage manifest in `src/intelligence_collector/storage/artifacts.py`; completion means Scope, Planning, Research, Synthesis, and Verification artifacts map to the VFS layout and preserve immutable versions and provenance (FR-010b, FR-021, SC-013).
- [X] T009 [P] Implement checkpoint persistence in `src/intelligence_collector/storage/checkpoints.py`; completion means typed graph state saves/loads atomically by `thread_id`, detects invalid latest checkpoints, and selects the latest valid checkpoint or returns a review-required error (FR-021, FR-022).
- [X] T010 [P] Implement local vector-index adapter in `src/intelligence_collector/storage/vector_index.py`; completion means raw source chunks index with source/artifact metadata and retrieval returns traceable chunk IDs, while index failure is explicit (FR-010a, FR-011, FR-012).
- [X] T011 [P] Implement structured event/error and audit contracts in `src/intelligence_collector/observability/events.py`; completion means failures identify stage/task, retryability, artifact refs, and resume/review guidance (FR-029).
- [X] T012 [P] Implement model router with role configuration, bounded retries, primary/fallback selection, and cost tier in `src/intelligence_collector/models/router.py`; completion means a forced primary failure selects the configured fallback exactly once within budget and emits a typed fallback event (plan model router; SC-011).
- [X] T013 [P] Implement Langfuse tracing adapter in `src/intelligence_collector/observability/tracing.py`; completion means graph nodes, MCP calls, model selection/fallback, artifact writes, verification, interrupts, and publication can emit correlated run/thread spans without making success depend on tracing availability (plan observability).
- [X] T014 [P] Implement input schema validation and prompt-injection/offensive/violent-content guardrail in `src/intelligence_collector/guardrails.py`; completion means rejected input returns a safe actionable `GuardrailDecision` and causes no graph, tool, checkpoint, VFS, or vector mutation (FR-002, User Story 2 scenario 4, plan guardrail).
- [X] T015 Define direct LangGraph typed state, node wrapper, checkpoint configuration, and interrupt plumbing in `src/intelligence_collector/graph/state.py` and `src/intelligence_collector/graph/build.py`; completion means all transitions are graph edges, state is serializable, and approval interrupt can resume by thread ID (FR-021, FR-024, FR-030).
- [X] T016 [P] Add contract tests for VFS, checkpoints, vector provenance, guardrail rejection, model fallback, and tracing in `tests/contract/test_storage_contracts.py`, `tests/contract/test_guardrail_contract.py`, `tests/contract/test_model_router_contract.py`, and `tests/contract/test_observability_contract.py`; completion means tests fail on path escape, invalid checkpoint, untraceable chunk, unsafe input, unbounded fallback, or missing correlation (constitution review gates).

## Phase 3: User Story 1 - Complete a Verified Research Run (Priority: P1) 🎯 MVP

**Goal**: Deliver one controlled-source, single-worker vertical slice from chat scope through verified report and approval-ready artifact.

**Independent Test**: Run `tests/integration/test_end_to_end_controlled.py` with fake MCP sources; assert scope, todos-before-retrieval, VFS evidence, indexing, ResearchComplete, synthesis, verification, UI state, and evidence-linked report (US1; SC-001–SC-005, SC-012).

### Tests for User Story 1

- [X] T017 [P] [US1] Add controlled-source end-to-end test in `tests/integration/test_end_to_end_controlled.py`; completion means a bounded request reaches Awaiting Approval with no live network, and asserts `write_todos` precedes every MCP call (US1 scenarios 1–5; FR-005, FR-006, SC-001, SC-002).
- [X] T018 [P] [US1] Add graph transition and state contract tests in `tests/contract/test_graph_transitions.py`; completion means invalid scope, failed todo write, incomplete research, and unverified publication cannot traverse their guarded edges (FR-003, FR-011, FR-024, FR-025).
- [X] T019 [P] [US1] Add Streamlit AppTest for progress and report display in `tests/ui/test_chat_progress.py`; completion means current stage, task progress, verification outcome, evidence links, and report availability render from graph snapshots (FR-023, SC-004).

### Implementation for User Story 1

- [X] T020 [US1] Implement Streamlit session/thread adapter in `src/intelligence_collector/app.py`; completion means chat turns map to a stable `thread_id`, UI reads graph snapshots, and UI code never writes authoritative artifacts (FR-001, FR-023).
- [X] T021 [US1] Implement Scope Agent node and clarification state in `src/intelligence_collector/agents/scope.py`; completion means objective, boundaries, constraints, success criteria, and deliverable format are assessed for explicit/testable consistency, broad scope receives narrowing options, and clear scope writes `scope/brief_summary.v1.json` then edges automatically to Planning (FR-003–FR-005, US2 scenarios 1–5).
- [X] T022 [US1] Implement `write_todos` tool in `src/intelligence_collector/tools/write_todos.py`; completion means it accepts only a non-empty list of `{todo_id, research_question, status=planned}`, persists `planning/todos.v1.json`, and blocks Research on any invalid or failed write (FR-006, FR-007, SC-002).
- [X] T023 [US1] Implement Planner node in `src/intelligence_collector/agents/planner.py`; completion means Planning invokes `write_todos` before any retrieval and exposes actionable failure without mutating Research state (FR-006, FR-029).
- [X] T024 [US1] Implement one isolated controlled-source MCP worker in `src/intelligence_collector/agents/worker.py` and MCP port in `src/intelligence_collector/tools/mcp.py`; completion means each task performs broad-to-narrow search, stops early or at its five-call hard limit, and writes raw capture, notes, excerpts, and evidence bundle VFS artifacts with provenance (FR-009–FR-010a, FR-009a–FR-009b, SC-011).
- [X] T025 [US1] Implement Supervisor and `think_tool` in `src/intelligence_collector/agents/supervisor.py` and `src/intelligence_collector/tools/think_tool.py`; completion means task status/progress assessment is persisted and `ResearchComplete` is emitted only when every todo is completed, required VFS files exist, and vector indexing succeeds (FR-008, FR-011, SC-012).
- [X] T026 [US1] Implement synthesis node in `src/intelligence_collector/agents/synthesis.py`; completion means indexed chunks/evidence are retrieved by provenance, previous and latest draft files are preserved, and no factual content is invented from chat history (FR-012, FR-010b).
- [X] T027 [US1] Implement verifier node and pass result in `src/intelligence_collector/agents/verification.py`; completion means scope coverage and claim-to-evidence checks produce a typed result, and passing output exposes a versioned verified artifact with navigable evidence refs (FR-013–FR-015, SC-003).
- [X] T028 [US1] Wire the MVP graph nodes and conditional edges in `src/intelligence_collector/graph/nodes.py` and `src/intelligence_collector/graph/routing.py`; completion means the controlled-source test executes Scope → Planning → Researching → ResearchComplete → Synthesis → Verification without a research-brief approval interrupt (FR-005, FR-024, SC-001).
- [X] T029 [US1] Add MVP runbook and fixture references in `tests/integration/README.md`; completion means a developer can reproduce the controlled-source flow using the quickstart command and inspect every stage artifact (quickstart; SC-013).

**Checkpoint**: MVP vertical slice is independently demonstrable before parallel workers or live MCP are added.

## Phase 4: User Story 2 - Clarify an Ambiguous Request (Priority: P1)

**Goal**: Make scope clarification safe, multi-turn, bounded, and automatically connected to Planning.

**Independent Test**: `tests/integration/test_scope_clarification.py` submits broad, incomplete, contradictory, and unsafe requests; it asserts questions/narrowing, no retrieval before clarity, and automatic Planning after boundaries are explicit (US2; FR-002–FR-005).

- [X] T030 [P] [US2] Add multi-turn scope tests in `tests/integration/test_scope_clarification.py`; completion means prior answers persist, only missing fields are requested, broad scopes receive narrowing suggestions, and no `write_todos`/MCP call occurs before scope completion (US2 scenarios 1–5).
- [X] T031 [US2] Harden Scope Agent clarification schema and narrowing suggestions in `src/intelligence_collector/agents/scope.py`; completion means all five scope acceptance scenarios pass and each accepted scope writes one durable brief summary (FR-003, FR-004, FR-005).
- [X] T032 [US2] Integrate guardrail and validation outcomes into Streamlit error rendering in `src/intelligence_collector/app.py` and `src/intelligence_collector/guardrails.py`; completion means unsafe/malformed input is actionable, safe, and mutation-free (FR-002, FR-029).

## Phase 5: User Story 3 - Recover from Failed Verification (Priority: P1)

**Goal**: Route typed BVF feedback to the smallest effective repair and terminate no-progress loops safely.

**Independent Test**: `tests/integration/test_bvf_routes.py` injects each issue type and repeated no-progress repairs; it asserts the correct route, preserved evidence, re-verification, bounded attempts, and `NEEDS_REVIEW` (US3; FR-016–FR-020, SC-005–SC-006).

- [X] T033 [P] [US3] Add verifier issue and BVF routing tests in `tests/integration/test_bvf_routes.py`; completion means `DRAFT_DEFECT`, `EVIDENCE_GAP`, and `PLAN_GAP` each route correctly and unknown types are rejected (FR-016, FR-017).
- [X] T034 [P] [US3] Add recovery-budget/no-progress tests in `tests/integration/test_recovery_limits.py`; completion means each repair re-verifies, preserves valid evidence, increments visible attempts, and exhaustion marks `NEEDS_REVIEW` (FR-018–FR-020, SC-006).
- [X] T035 [US3] Implement typed verifier issues and deterministic BVF routing in `src/intelligence_collector/agents/verification.py` and `src/intelligence_collector/graph/routing.py`; completion means exactly one primary issue type selects draft revision, targeted research, or re-planning and a progress fingerprint blocks no-op loops (FR-016, FR-017).
- [X] T036 [US3] Implement draft revision, targeted-gap research, and affected-task re-planning nodes in `src/intelligence_collector/agents/synthesis.py`, `src/intelligence_collector/agents/worker.py`, and `src/intelligence_collector/agents/planner.py`; completion means valid evidence is reused, changed artifacts invalidate prior verification, and every repaired candidate re-enters full verification (FR-017, FR-018).
- [X] T037 [US3] Add `NEEDS_REVIEW` UI and artifact persistence in `src/intelligence_collector/app.py` and `src/intelligence_collector/storage/artifacts.py`; completion means unresolved feedback is visible and no UI/output labels the artifact verified (FR-020, FR-023).

## Phase 6: User Story 4 - Resume an Interrupted Run (Priority: P2)

**Goal**: Restore durable state and continue without repeating completed retrieval.

**Independent Test**: `tests/integration/test_resume.py` interrupts at each major stage, reloads the same thread, and asserts valid checkpoint restoration, intact artifacts, and no duplicate completed MCP calls (US4; FR-021–FR-022, SC-007).

- [X] T038 [P] [US4] Add interruption/resume integration tests in `tests/integration/test_resume.py`; completion means Scope, Planning, Researching, Synthesis, and Verifying interruption points restore expected state and invalid checkpoints produce the documented recoverable error or `NEEDS_REVIEW` (US4 scenarios 1–3).
- [X] T039 [US4] Integrate checkpoint save/load boundaries into every graph node in `src/intelligence_collector/graph/build.py` and `src/intelligence_collector/storage/checkpoints.py`; completion means state includes scope, todos, artifacts, assessments, drafts, verification history, counters, and publication state (FR-021).
- [X] T040 [US4] Add idempotent task/artifact resume guards in `src/intelligence_collector/agents/worker.py` and `src/intelligence_collector/storage/artifacts.py`; completion means completed retrieval is skipped unless incomplete, invalidated, or explicitly repaired (FR-022, SC-007).

## Phase 7: User Story 5 - Control Publication (Priority: P2)

**Goal**: Bind human approval to an unchanged verified artifact and publish atomically.

**Independent Test**: `tests/integration/test_approval_binding.py` rejects, approves, mutates, and retries publication; it asserts no unauthorized permanent write and exact hash equality on success (US5; FR-025–FR-028a, SC-008–SC-009).

- [X] T041 [P] [US5] Add approval/rejection and stale-hash tests in `tests/integration/test_approval_binding.py`; completion means unverified, failed, `NEEDS_REVIEW`, stale, rejected, and write-failure cases leave permanent output unchanged (US5 scenarios 1–5).
- [X] T042 [US5] Implement approval interrupt/resume UI in `src/intelligence_collector/app.py` and `src/intelligence_collector/publication.py`; completion means only verified artifacts expose approval, decisions bind actor/time/hash, and rejection is durable without publication (FR-025–FR-027).
- [X] T043 [US5] Implement atomic publisher in `src/intelligence_collector/publication.py`; completion means it is the only permanent-output writer, validates unchanged approved hash and verification ID, records success/failure, and never exposes partial output (FR-028, FR-028a, SC-008–SC-009).

## Phase 8: Cross-Cutting Resilience, Observability, and Acceptance

**Purpose**: Add explicit demonstrations and opt-in external validation after the controlled vertical slice works.

- [X] T044 [P] Add model fallback integration test and trace assertions in `tests/integration/test_model_fallback.py`; completion means forced primary timeout selects fallback within budget and Langfuse records both attempts (plan model router; SC-011).
- [X] T045 [P] Add Langfuse trace contract assertions in `tests/integration/test_langfuse_tracing.py`; completion means run, node, MCP, artifact, verification, interrupt, and publication spans share run/thread correlation, while tracing outage does not fabricate success (plan observability).
- [X] T046 [P] Add repeatable Ragas faithfulness evaluation in `tests/evaluation/test_ragas_faithfulness.py` using controlled evidence contexts; completion means the fixed dataset produces a recorded score/configuration and evaluation artifacts remain separate from publication status (plan Ragas; SC-003).
- [X] T047 [P] Add opt-in live MCP acceptance test in `tests/integration/test_live_mcp_opt_in.py`; completion means the test is skipped unless an explicit environment flag and configured MCP source are present, and when enabled it verifies provenance, five-call budgets, VFS artifacts, and vector indexing (FR-009–FR-011, SC-011–SC-013).
- [X] T048 Run the complete quickstart validation in `specs/001-verified-research-report/quickstart.md`; completion means controlled end-to-end, ambiguity, resume, fallback, all BVF routes, guardrail, faithfulness, and approval-binding demonstrations pass before any optimization work (all acceptance criteria).
- [X] T049 [P] Add operational documentation for VFS layout, checkpoint recovery, model configuration, Langfuse opt-in, Ragas evaluation, and live MCP opt-in in `docs/deep-research-runbook.md`; completion means a new developer can reproduce controlled tests without credentials and understands permanent-output safeguards (constitution; plan contracts).
- [X] T050 Review and harden security/cost boundaries in `src/intelligence_collector/config.py`, `src/intelligence_collector/guardrails.py`, `src/intelligence_collector/storage/vfs.py`, and `src/intelligence_collector/models/router.py`; completion means secrets never enter artifacts/logs, path escapes fail, search/recovery/model budgets are finite, and no Deep Agents SDK is present (constitution; FR-002, FR-019, FR-028a).

## Phase 9: Convergence

**Purpose**: Close gaps found by comparing the current implementation with the approved specification,
plan, constitution, and existing tasks. These tasks are append-only remediation work.

- [X] T051 Implement the executable typed LangGraph workflow with concrete nodes, conditional edges, durable checkpoint integration, and approval interrupt in `src/intelligence_collector/graph/build.py`, `src/intelligence_collector/graph/nodes.py`, and `src/intelligence_collector/graph/state.py`; completion means the graph executes Scope → Planning → Researching → ResearchComplete → Synthesis → Verification and resumes by thread ID without bypass paths (FR-030; Constitution II; missing/CRITICAL).
- [X] T052 Implement multi-turn Scope Agent state, bounded-scope narrowing suggestions, brief-summary VFS persistence, and automatic Planning transition in `src/intelligence_collector/agents/scope.py` and `src/intelligence_collector/graph/nodes.py`; completion means no retrieval or todo write occurs before objective, boundaries, constraints, success criteria, and format are explicit/testable (FR-003–FR-005; partial/HIGH).
- [X] T053 Complete the MCP worker and Supervisor research stage in `src/intelligence_collector/agents/worker.py`, `src/intelligence_collector/agents/supervisor.py`, and `src/intelligence_collector/tools/mcp.py`; completion means workers perform broad-to-narrow searches with 3/5-call budgets, persist notes/excerpts/evidence, and Supervisor persists assessment and validates ResearchComplete prerequisites (FR-009–FR-011; partial/HIGH).
- [X] T054 Implement stage artifact repository and manifest in `src/intelligence_collector/storage/artifacts.py`; completion means Scope brief, Planning todos, per-task raw/notes/excerpts/evidence, progress assessment, previous/latest drafts, and verification reports are immutable, versioned, and addressable (FR-010b, FR-021, SC-013; missing/MEDIUM).
- [X] T055 Implement evidence-backed synthesis in `src/intelligence_collector/agents/synthesis.py`; completion means indexed retrieval returns source IDs, previous/latest drafts are persisted through the artifact repository, and claims cannot be produced without retrievable evidence context (FR-012, FR-015; partial/HIGH).
- [X] T056 Implement typed verification checks, progress fingerprints, bounded BVF recovery, and `NEEDS_REVIEW` terminal routing in `src/intelligence_collector/agents/verification.py`, `src/intelligence_collector/graph/routing.py`, and `src/intelligence_collector/graph/nodes.py`; completion means DRAFT_DEFECT, EVIDENCE_GAP, and PLAN_GAP select their required repair and no-progress loops terminate (FR-016–FR-020; partial/HIGH).
- [X] T057 Integrate checkpoint save/load and idempotent resume guards across all graph nodes in `src/intelligence_collector/storage/checkpoints.py`, `src/intelligence_collector/graph/build.py`, and `src/intelligence_collector/agents/worker.py`; completion means interruption at every major stage restores state and does not repeat completed retrieval (FR-021–FR-022, SC-007; missing/HIGH).
- [X] T058 Implement approval interrupt, review UI, stale-hash rejection, and publication-state transitions in `src/intelligence_collector/app.py` and `src/intelligence_collector/publication.py`; completion means only an unchanged verified artifact can be approved and atomically written to permanent output (FR-025–FR-028a; partial/HIGH).
- [X] T059 Implement structured error/audit contracts and correlated event emission in `src/intelligence_collector/observability/events.py`; completion means every stage/task failure exposes retry/resume/review guidance and preserves artifact references (FR-029; missing/MEDIUM).
- [X] T060 Implement model-router role configuration, bounded retry/fallback events, and Langfuse tracing adapter in `src/intelligence_collector/models/router.py` and `src/intelligence_collector/observability/tracing.py`; completion means primary failure falls back once within budget and all node/tool/model/artifact/approval events share run/thread correlation (plan observability; missing/HIGH).
- [X] T061 Implement the full Streamlit run-status UI in `src/intelligence_collector/app.py`; completion means the UI displays stage, todo progress, supervisor assessment, verification outcome, recovery state, evidence links, resume errors, and approval controls without writing authoritative state (FR-023; missing/HIGH).
- [X] T062 Add repeatable controlled-source Ragas faithfulness evaluation in `tests/evaluation/test_ragas_faithfulness.py`; completion means the fixed dataset records a reproducible score/configuration without changing publication status (SC-003; missing/HIGH).
- [X] T063 Add stage-level interruption tests and opt-in live MCP acceptance in `tests/integration/test_resume.py` and `tests/integration/test_live_mcp_opt_in.py`; completion means tests verify checkpoint restoration, provenance, search budgets, VFS artifacts, and vector indexing, while live tests skip without explicit configuration (SC-007, SC-011–SC-013; missing/MEDIUM).
- [X] T064 Add the operational runbook and execute the complete quickstart validation in `docs/deep-research-runbook.md` and `specs/001-verified-research-report/quickstart.md`; completion means controlled end-to-end, ambiguity, fallback, BVF, guardrail, evaluation, resume, and approval demonstrations are reproducible locally (all acceptance criteria; missing/MEDIUM).

## Phase 10: LLM Decision Audit and Convergence

- [X] T065 Implement versioned prompts and typed LLM decision schemas for Scope, Planner, Supervisor, Worker, Synthesizer, Verifier, and Fixer in `src/intelligence_collector/agents/prompts.py` and `src/intelligence_collector/contracts.py`; completion means every semantic role has a named prompt version and validated response model (FR-031).
- [X] T066 Implement explicit live/fake model modes, structured parsing, fallback/timeouts/retry settings, and secret-safe decision tracing in `src/intelligence_collector/models/router.py` and `src/intelligence_collector/observability/tracing.py`; completion means live mode fails loudly without a model, fake mode is opt-in, and every decision records role/prompt version/model/fallback metadata (FR-032).
- [X] T067 Implement real Scope Agent LLM assessment and same-thread clarification/Planning transition in `src/intelligence_collector/agents/scope.py`, `src/intelligence_collector/graph/build.py`, and `src/intelligence_collector/app.py`; completion means fake-model tests cover ambiguous and clear requests and live smoke is explicit opt-in (FR-033, US2).
- [X] T068 Add semantic-role LLM audit tests and opt-in live smoke documentation in `tests/integration/test_scope_llm.py`, `tests/integration/test_scope_live_smoke.py`, and `docs/deep-research-runbook.md`; completion means no role claims LLM support without a real call path, and offline tests never silently use fakes (FR-031–FR-033).

## Phase 11: LangChain LLM Architecture

- [X] T069 Implement the central LangChain `ModelFactory` and role router using
  `langchain.chat_models.init_chat_model` in `src/intelligence_collector/models/factory.py` and
  `src/intelligence_collector/models/router.py`; completion means every role has primary/fallback,
  timeout, reasoning settings, explicit live/fake mode, and secret-safe trace metadata (FR-032,
  FR-034).
- [X] T070 Define versioned `ChatPromptTemplate` objects with `MessagesPlaceholder` and data
  variables for Scope, Planner, Supervisor, Worker, Synthesizer, Verifier, and Fixer in
  `src/intelligence_collector/agents/prompts.py`; completion means no role prompt is embedded in UI
  callbacks or graph routing and user/evidence text is passed as data (FR-035).
- [X] T071 Integrate LangChain `SystemMessage`, `HumanMessage`, and `AIMessage` history into the
  two-node Scope flow (`clarify_with_user` and `write_research_brief`) in
  `src/intelligence_collector/agents/scope.py`, `src/intelligence_collector/graph/state.py`, and
  `src/intelligence_collector/graph/build.py`; completion means ambiguous input pauses with one
  focused question, a clarification resumes the same thread, the brief is created by a second
  structured call, and clear scope automatically edges to Planning (FR-033–FR-035).
- [X] T072 Add bounded structured-output parsing/retry/error handling for every semantic role in
  `src/intelligence_collector/models/router.py` and `src/intelligence_collector/agents/llm_roles.py`;
  completion means invalid output cannot become `SCOPE_CLEAR` or `VERIFIED` and produces a traced
  error (FR-036).
- [X] T073 Add offline fake-model tests and opt-in live LangChain smoke tests demonstrating ambiguous
  then clarified and clear Scope requests, including prompt version, model identifier, validated
  output, graph edge, and trace assertions in `tests/integration/test_langchain_scope.py`;
  completion means offline tests pass without credentials and live tests run only with explicit
  configuration (FR-033–FR-036).

## Dependencies & Execution Order

### Phase Dependencies

- Phase 1 Setup has no dependencies.
- Phase 2 Foundational depends on Phase 1 and blocks all user stories.
- Phase 3 US1 depends on Phase 2 and is the required MVP vertical slice.
- Phase 4 US2 depends on the foundational guardrail/graph contracts and may reuse US1 graph/UI infrastructure.
- Phase 5 US3 depends on US1 verification/synthesis and may reuse US1 evidence fixtures.
- Phase 6 US4 depends on US1 graph/checkpoint integration.
- Phase 7 US5 depends on US1 verified artifact and graph interrupt.
- Phase 8 depends on the controlled US1 flow plus the relevant story phase; live MCP is last and opt-in.

### User Story Completion Order

`US1 → (US2, US3, US4, US5 in parallel where staffed) → Phase 8 live/quality acceptance`.

## Parallel Execution Examples

After Phase 2: T017–T019 can be prepared in parallel; implementation T020–T027 is sequential where
contracts require it. After US1: T030, T033–T034, T038, and T041 can run in parallel because they
use separate test files/fixtures; their implementation tasks merge only at graph routing and UI
boundaries. Phase 8 tasks T044–T047 are independent after their prerequisite story contracts pass.

## Implementation Strategy

1. Complete Setup and Foundational phases.
2. Implement and demonstrate US1 with controlled sources end to end.
3. Stop at the MVP checkpoint and validate all stage artifacts, evidence links, and UI progress.
4. Add scope hardening, BVF recovery, resume, and approval as independently testable increments.
5. Add fallback, Langfuse, Ragas, and the opt-in live MCP run only after controlled acceptance passes.
6. Reuse fixtures, ports, contracts, and persisted artifact helpers at every later phase; do not add
   parallel workers or optimization until the single-worker flow is proven.

## Phase 12: Convergence

- [X] T074 Route Planner, Supervisor, Worker, Synthesizer, Verifier, and Fixer through their
  versioned `ChatPromptTemplate` objects and `MessagesPlaceholder`/data variables in
  `src/intelligence_collector/agents/llm_roles.py`; completion means each role renders its prompt,
  invokes the LangChain message path, and validates its typed output (FR-035; partial/HIGH).
- [X] T075 Add bounded structured-output parse retries and secret-safe validation-error tracing in
  `src/intelligence_collector/models/router.py`; completion means malformed live responses are
  retried within the configured budget, recorded with role/prompt/model metadata, and never become
  a successful workflow decision (FR-036; partial/HIGH).
- [X] T076 Integrate the Scope Agent graph and same-thread conversation state into
  `src/intelligence_collector/app.py`; completion means Streamlit submits validated turns to the
  LangGraph thread, displays the focused clarification, and automatically enters Planning after a
  clear brief instead of using deterministic scope-only logic (FR-001, FR-033; missing/HIGH).
- [X] T077 Declare the `langchain` runtime dependency required by
  `langchain.chat_models.init_chat_model` in `pyproject.toml`; completion means a clean environment
  can import the configured model factory without relying on an undeclared transitive package
  (FR-034; partial/HIGH).
- [X] T078 Add a checkpoint-backed Scope clarification resume test using the same LangGraph
  `thread_id` and persisted `SystemMessage`/`HumanMessage`/`AIMessage` turns; completion means an
  interrupted clarification resumes without losing prior turns or repeating completed decisions
  (FR-021, FR-033; partial/MEDIUM).
- [X] T079 Inject the run-scoped VFS into the Scope graph's `write_research_brief` node and persist
  `scope/brief_summary.v1.json`; completion means a clear Scope edge cannot reach Planning unless the
  brief artifact is durably written (FR-010b, FR-033a; partial/MEDIUM).
- [X] T080 Replace the unconditional skip in `tests/integration/test_scope_live_smoke.py` with an
  explicit opt-in test that constructs the configured model through `init_chat_model`, invokes the
  Scope prompt, and asserts validated output and graph routing; completion means it skips only when
  live configuration is absent and otherwise exercises a real request (FR-031–FR-033; partial/MEDIUM).

## Phase 13: Two-Stage Scope Completion

- [X] T081 [P] [US2] Define and validate a dedicated `ResearchBrief` contract plus decision-specific
  `ScopeAssessment` validation in `src/intelligence_collector/contracts.py`; completion means malformed
  clarification/clear results and empty brief fields cannot route the graph (FR-033b, FR-036).
- [X] T082 [P] [US2] Refine the versioned Scope prompt templates in
  `src/intelligence_collector/agents/prompts.py` and `src/intelligence_collector/agents/scope.py`;
  completion means actual Human/AI history is passed once through `MessagesPlaceholder`, brief output
  uses `ResearchBrief`, and constraints are never invented (FR-033a–FR-033c).
- [X] T083 [US2] Implement the conditional clarification, brief, and Planning nodes in
  `src/intelligence_collector/graph/build.py`; completion means clarification ends with exactly one
  AI question, clear scope writes the brief, Planning consumes it and calls `write_todos`, and no
  retrieval edge is enabled before the todo artifact exists (FR-006, FR-033a–FR-037).
- [X] T084 [US4] Configure durable Scope graph checkpoint/VFS dependencies in
  `src/intelligence_collector/app.py` and `pyproject.toml`; completion means a recreated graph can
  resume the same thread and authoritative brief/todo artifacts survive process restart (FR-021).
- [X] T085 [P] [US2] Add clear, multi-turn, automatic-transition, fallback, malformed-output, and
  total-model-failure coverage in `tests/integration/test_langchain_scope.py`; completion means model
  calls, prompts, state updates, edges, and failure behavior are asserted (FR-033–FR-037).
- [X] T086 [P] [US4] Add process-restart Scope resume coverage in
  `tests/integration/test_scope_checkpoint.py`; completion means a new graph/checkpointer instance
  resumes the same thread without losing Human/AI turns (FR-021, FR-033c).
- [X] T087 Run the full test and lint suites and record the completed Phase 13 tasks in
  `specs/001-verified-research-report/tasks.md`; completion means all controlled tests pass and the
  opt-in live test remains the only credential-dependent Scope test.

## Phase 14: Single-Question Research Brief

- [X] T088 [US2] Replace the multi-field `ResearchBrief` with one validated detailed
  `research_question` in `src/intelligence_collector/contracts.py` and update the transformation
  prompt in `src/intelligence_collector/agents/prompts.py` (FR-033b).
- [X] T089 [US2] Persist and route the single research question through Scope and Planning in
  `src/intelligence_collector/agents/scope.py` and `src/intelligence_collector/graph/build.py`;
  completion means `write_todos` receives exactly that question (FR-037).
- [X] T090 [P] [US2] Update Scope integration fixtures and assertions in
  `tests/integration/test_langchain_scope.py`, `tests/integration/test_scope_checkpoint.py`, and
  `tests/integration/test_scope_llm.py` for the single-question contract.
- [X] T091 Run the full pytest and Ruff suites and mark Phase 14 complete in
  `specs/001-verified-research-report/tasks.md`.

## Phase 15: Scope Verification Message

- [X] T092 [US2] Add decision-specific `verification` validation to `ScopeAssessment` and document
  assistant-summary authority rules in `src/intelligence_collector/contracts.py` and
  `src/intelligence_collector/agents/prompts.py` (FR-033c).
- [X] T093 [US2] Append exactly one verification `AIMessage` for clear scope while preserving the
  automatic brief edge in `src/intelligence_collector/graph/build.py` (FR-033a, FR-033c).
- [X] T094 [P] [US2] Update clear, multi-turn, and checkpoint-resume message-state assertions in
  `tests/integration/test_langchain_scope.py`, `tests/integration/test_scope_checkpoint.py`, and
  `tests/integration/test_scope_llm.py`.
- [X] T095 Run pytest, Ruff, and diff validation and mark Phase 15 complete in
  `specs/001-verified-research-report/tasks.md`.

## Phase 16: High-Fidelity Langfuse Hierarchy

- [X] T096 [P] [US1] Add required live Langfuse configuration and secret-safe defaults in
  `src/intelligence_collector/config.py` and `.env.example` (FR-038).
- [X] T097 [US1] Implement root execution traces, thread/session propagation, graph-node spans,
  router validation events, LangChain generation callbacks, and flush handling in
  `src/intelligence_collector/observability/tracing.py`, `src/intelligence_collector/models/router.py`,
  and `src/intelligence_collector/graph/build.py` (FR-038).
- [X] T098 [US1] Wire required tracing around every Streamlit research submission in
  `src/intelligence_collector/app.py`; completion means missing configuration fails before model use
  and every result/error flushes one correlated trace (FR-038).
- [X] T099 [P] [US1] Add credential-free hierarchy, correlation, callback, flush, failure, and secret
  redaction tests in `tests/contract/test_observability_contract.py` and
  `tests/integration/test_langfuse_tracing.py` (FR-038).
- [X] T100 Update `docs/deep-research-runbook.md`, run offline pytest/Ruff/diff validation, and mark
  Phase 16 complete in `specs/001-verified-research-report/tasks.md`.
- [X] T101 [US1] Record complete rendered model prompts and validated JSON responses as textual
  Langfuse inputs/outputs, keep hashes and prompt versions as metadata only, strengthen the Scope and
  Brief instructions, and cover the visible payload contract in tracing tests (FR-038a).
- [X] T102 [US2] Make the live SQLite checkpoint path self-initializing and actionable on invalid
  paths or permission failures; use a SQLite-specific default and cover parent creation and failure
  reporting in unit tests (FR-021, FR-029).
- [X] T103 [US1] Use the canonical Langfuse base URL setting, trim configuration values, and perform
  a fail-fast authentication check before live model execution so invalid project/region key pairs
  produce one actionable error instead of repeated unauthorized batch-export failures (FR-038).

## Phase 17: Convergence

- [X] T104 Replace the current multi-field Scope decision with the exact three-field
  `need_clarification`, `question`, and `verification` contract; update the versioned clarification
  prompt, Scope Agent, graph routing, brief input, and model fixtures while preserving the separate
  conversation-derived `ResearchBrief` call in `src/intelligence_collector/contracts.py`,
  `src/intelligence_collector/agents/prompts.py`, `src/intelligence_collector/agents/scope.py`, and
  `src/intelligence_collector/graph/build.py` per FR-033 and FR-033a (contradicts).
- [X] T105 Refactor the Streamlit presentation in `src/intelligence_collector/app.py` into a
  chat-first transcript rendered with user/assistant chat messages, a compact active status line,
  and a collapsed-by-default `Details` panel for agent, tool, task, verification, recovery, and
  evidence metadata; ensure each completed turn adds at most one plain-Markdown assistant message
  and never places structured JSON or internal metadata in the transcript per FR-023 and FR-023c
  (contradicts).
- [X] T106 Connect Streamlit progress rendering to graph-node and approved-tool lifecycle events so
  each stable activity item transitions from running to success or failure during execution, the
  processing indicator animates only while backend work is active, and clarification pause renders
  a static `Waiting for your response` state in `src/intelligence_collector/app.py`,
  `src/intelligence_collector/graph/build.py`, and `src/intelligence_collector/graph/nodes.py` per
  FR-023a and FR-023b (partial).
- [X] T107 Add and update credential-free Scope and Streamlit acceptance coverage in
  `tests/integration/test_langchain_scope.py`, `tests/integration/test_scope_llm.py`, and
  `tests/ui/test_chat_progress.py` for the exact three-field clarification schema, separate brief
  generation, ChatGPT-style transcript, one-assistant-message limit, collapsed Details panel,
  realtime lifecycle updates, and static clarification waiting state per SC-004 and SC-004a
  (missing).

## Phase 18: Graph Maintainability

- [X] T108 Refactor graph nodes and edges for readability: move `node_wrapper` to
  `src/intelligence_collector/graph/instrumentation.py`; add shared `Node` names, `ScopeNodes`,
  `approval`, and `stage_only` to `src/intelligence_collector/graph/nodes.py`; add named
  `route_after_clarify` and `route_after_verify` routers (null-safe verification) to
  `src/intelligence_collector/graph/routing.py`; reduce `src/intelligence_collector/graph/build.py`
  to documented wiring; and key `_NODE_ACTIVITY` in `src/intelligence_collector/app.py` by `Node`.
  Completion means the existing suite passes unchanged and `build_graph`/`node_wrapper` imports
  remain compatible (FR-030, FR-030a).
- [X] T109 [US1] Implement the LLM-backed Planner Agent: change `TodoPlan` to a questions-only
  contract (1–5 standalone, non-blank, unique) in `src/intelligence_collector/contracts.py`; add the
  versioned `planner.v2` prompt in `src/intelligence_collector/agents/prompts.py`; add
  `PlannerAgent.define_questions`/`write_todos` and rename `plan` to `create_todos` in
  `src/intelligence_collector/agents/planner.py`; move the Planning node from `ScopeNodes` to
  `PlannerNodes` and inject `planner_agent` in `src/intelligence_collector/graph/nodes.py` and
  `src/intelligence_collector/graph/build.py`. Completion means Planning persists one planned todo
  per LLM-defined question, invalid planner output writes no todo list, and tracing shows a
  `model.validation.planner` event, covered in `tests/integration/test_langchain_scope.py`,
  `tests/integration/test_langfuse_tracing.py`, and
  `tests/contract/test_semantic_decision_contracts.py` (FR-006, FR-031, FR-037, FR-037a).
- [X] T110 [US1] Refactor research into an LLM tool-calling sub-agent and a delegating Supervisor:
  add `ResearchAgent` (LangGraph `llm_call` → `tool_node` → `compress_research` with
  `route_after_llm`, `tavily_search` tool, `researcher.v1`/`research-compress.v1` prompts, fresh MCP
  client and hard per-task budget per sub-agent) in `src/intelligence_collector/agents/researcher.py`;
  add `ResearchFindings` in `src/intelligence_collector/contracts.py`; add
  `ModelRouter.invoke_with_tools` with thread-local call metadata in
  `src/intelligence_collector/models/router.py`; replace the rule-based search loop with
  `persist_research`/`run_worker(…, researcher)` in `src/intelligence_collector/agents/worker.py`; add
  `ResearchSupervisor` (bounded parallel delegation, plan-order persistence, sibling-safe failures)
  in `src/intelligence_collector/agents/supervisor.py`; add `think_tool` blockers; remove the
  hard-coded Tavily `sufficient` flag and redact MCP errors in
  `src/intelligence_collector/tools/mcp.py`; add `INTELLIGENCE_MAX_CONCURRENT_RESEARCHERS` in
  `src/intelligence_collector/config.py`; wire `ResearchNodes(vfs, index, researcher)` in
  `src/intelligence_collector/graph/nodes.py` and `src/intelligence_collector/app.py`. Completion
  means per-task budgets no longer share one global MCP counter, insufficient findings complete the
  task with recorded gaps, and `tests/integration/test_worker_mcp.py` covers broad→narrow search,
  early stop, hard budget, redacted MCP failure, parallel delegation, and sibling-safe failure
  (FR-008, FR-008a, FR-009–FR-009d, FR-010, FR-010a, FR-011, FR-031).

## Phase 19: Mandatory Build-Verify-Fix Loop

- [X] T111 [US3] Implement the deterministic source-verification script with ten report-vs-evidence
  checks, stable `[S<n>]` source catalog, failure fingerprint, and standalone CLI in
  `src/intelligence_collector/source_verification.py`; add `VerificationCheck`, per-check
  `check_results`, and `Reflection` contracts in `src/intelligence_collector/contracts.py` (FR-014,
  FR-015, FR-016, FR-016a).
- [X] T112 [US3] Implement the LLM Report Writer (`report-writer.v1`) for Build and Fix drafts and the
  BVF reflection (`bvf-reflect.v1`) with accumulating lesson memory in
  `src/intelligence_collector/agents/report_writer.py` and `src/intelligence_collector/agents/prompts.py`
  (FR-017b, FR-031).
- [X] T113 [US3] Wire the BVF loop in `src/intelligence_collector/graph/nodes.py` (`BVFNodes`:
  `synthesizing`, `verifying`, `fixing`, `researching_gaps`, `needs_review`),
  `src/intelligence_collector/graph/routing.py` (`route_after_verify`, `route_after_fix`), and
  `src/intelligence_collector/graph/build.py`; add BVF state to
  `src/intelligence_collector/graph/state.py`; wire `recovery_budget` and BVF activity/lessons in
  `src/intelligence_collector/app.py`. Completion means only a fully passing report reaches
  approval, DRAFT_DEFECT reflects then redrafts, EVIDENCE_GAP researches each gap once then requires
  disclosure, and budget exhaustion or two no-progress repeats end in `NEEDS_REVIEW` (FR-017,
  FR-017a, FR-018, FR-019, FR-020).
- [X] T114 [P] [US3] Add `tests/unit/test_source_verification.py` (every check, primary-issue
  precedence, fingerprint) and `tests/integration/test_bvf_loop.py` (reflect-and-fix with lessons in
  the fix prompt, no-progress escalation, budget exhaustion, evidence-gap research and disclosure,
  standalone script verdict) (SC-005, SC-006).

## Phase 20: URL Deduplication and Documentation Language

- [X] T115 [US1] Track retrieved URLs by normalized identity: add `normalize_url` and the run-level
  `UrlRegistry` (`research/url_registry.json`) in `src/intelligence_collector/tools/url_tracking.py`;
  hide already-retrieved pages from later search results inside a sub-agent and dedupe sources by
  URL in `src/intelligence_collector/agents/researcher.py` (prompt `researcher.v2`); reuse the first
  capture instead of re-writing and re-indexing in `src/intelligence_collector/agents/worker.py`
  and `src/intelligence_collector/agents/supervisor.py`; share URL normalization with
  `src/intelligence_collector/source_verification.py`. Covered by `tests/unit/test_url_tracking.py`
  and `tests/integration/test_worker_mcp.py` (FR-010, FR-010a, FR-010c).
- [X] T116 Translate the Vietnamese Clarifications (Session 2026-09-25) in
  `specs/001-verified-research-report/spec.md` to English; all specification documents are English.

## Phase 21: Embedding-Based Evidence Retrieval for the Build/Fix Prompt

- [X] T117 [US1] Add a shared `VectorIndex` interface and a real, locally persisted embedding
  backend: define the `VectorIndex` Protocol and extend `LocalVectorIndex.retrieve()` with a
  `todo_id` scope in `src/intelligence_collector/storage/vector_index.py`; add `ChromaVectorIndex`
  (OpenAI embeddings via `langchain-openai`, local persistence via `chromadb`, an injectable
  `embeddings` for offline testing) in
  `src/intelligence_collector/storage/chroma_vector_index.py`; add `chromadb`/`langchain-openai` to
  `pyproject.toml` and `chroma_path`/`embedding_model`/`evidence_top_k` to
  `src/intelligence_collector/config.py` and `.env.example` (FR-012a).
- [X] T118 [US1] Narrow the Build/Fix prompt to each research question's most relevant evidence
  without changing what Verify checks against: add `select_relevant_evidence` (per-bundle top-K,
  never global; retrieval failure leaves a bundle unfiltered) in
  `src/intelligence_collector/agents/report_writer.py`; call it from `BVFNodes.synthesizing` only —
  `BVFNodes._catalog` (used by `verifying` and `researching_gaps`) stays on the full, unfiltered
  catalog — in `src/intelligence_collector/graph/nodes.py`; wire `ChromaVectorIndex` and
  `evidence_top_k` into the live app in `src/intelligence_collector/app.py`. Completion means a
  bundle at or under `evidence_top_k` is unchanged, a larger bundle is narrowed to exactly
  `evidence_top_k` sources scoped to its own question, and the full evidence catalog persisted for
  a run is unaffected by this narrowing, covered in `tests/unit/test_vector_index.py`,
  `tests/unit/test_evidence_selection.py`, and
  `tests/integration/test_bvf_loop.py::test_evidence_selection_narrows_the_build_prompt_but_not_verification`
  (FR-012, FR-012a).

## Phase 22: Gemini Primary with GPT Fallback

- [X] T119 [US1] Configure the live app's shared model router with a Gemini primary and an
  optional GPT fallback: add `langchain-google-genai` to `pyproject.toml` (`ModelFactory.create`
  already forwards any `model_provider` string to `init_chat_model` unchanged, so no factory code
  change was needed); wire `SCOPE_FALLBACK_MODEL`/`SCOPE_FALLBACK_PROVIDER` in
  `src/intelligence_collector/app.py::_get_scope_agent` (`ModelRouter(primary, fallback, ...)`, an
  unset fallback model preserves prior single-model behavior); update `SCOPE_MODEL`/`SCOPE_PROVIDER`
  defaults to Gemini and document the new fallback vars in `.env.example`. Completion means every
  role sharing the Scope Agent's router (Planner, Researcher, Report Writer, Fixer, via their
  default constructors in `graph/build.py` and `graph/nodes.py`) gets the same primary/fallback pair
  with no per-role changes, covered offline (mocked `init_chat_model`) in
  `tests/unit/test_scope_agent_wiring.py` (FR-031, FR-032, FR-032a). Superseded by Phase 23, which
  replaces the single shared router with per-role reasoning/standard tiers.

## Phase 23: Role-Tiered Model Routing and Acceptance-Criteria Verification

- [X] T120 [US1] Extend `TodoPlan` with `acceptance_criteria` (1–5 non-blank, non-duplicate
  strings) in `src/intelligence_collector/contracts.py`; bump the Planner prompt to `planner.v3` in
  `src/intelligence_collector/agents/prompts.py` to also require acceptance criteria; change
  `PlannerAgent.define_questions`/`write_todos` to `define_plan`/`write_todos` (now returning
  `tuple[list[ResearchTodo], list[str]]` and persisting `planning/acceptance_criteria.v1.json`) in
  `src/intelligence_collector/agents/planner.py`; thread `acceptance_criteria` through
  `PlannerNodes.planning` and `GraphState`/`ResearchGraphState`. Completion means invalid planner
  output (missing or malformed criteria, same as malformed questions) never writes a todo list,
  covered in `tests/contract/test_semantic_decision_contracts.py` and
  `tests/integration/test_langchain_scope.py` (FR-006a, FR-037a).
- [X] T121 [US1] Add the LLM acceptance-criteria check: `AcceptanceVerifier` (role `verify`,
  `acceptance.v1` prompt) and `merge_acceptance_verdict` in
  `src/intelligence_collector/agents/acceptance_verifier.py`; wire an optional
  `acceptance_verifier` into `ResearchNodes`/`BVFNodes` in
  `src/intelligence_collector/graph/nodes.py` so `verifying()` runs it only when the deterministic
  `verify_report` already passed, folding any unmet criterion into `VerificationResult` as an
  `acceptance_criteria` check with `IssueType.DRAFT_DEFECT` — reusing the existing Fix/reflect loop
  with no new routing. Completion means a structurally failing draft never reaches this (paid) LLM
  call and an unmet criterion is fixed like any other `DRAFT_DEFECT`, covered in
  `tests/unit/test_acceptance_verifier.py` and `tests/integration/test_bvf_loop.py` (FR-016b).
- [X] T122 [US1] Replace the single shared model router with per-role tiers: add
  `reasoning_fallback_model/provider` and `standard_fallback_model/provider` to
  `RoleModelSettings.from_env()` in `src/intelligence_collector/config.py`; give
  `ModelFactory.create_role_router` a `callbacks` param and per-tier fallback construction in
  `src/intelligence_collector/models/factory.py`; update `DEEP_REASONING_ROLES`/`STANDARD_ROLES` to
  the real live role names (`scope`, `planner`, `verify` / `researcher`, `research_compressor`,
  `synthesizer`, `fixer`) and add `RoleModelRouter.invoke_with_tools` in
  `src/intelligence_collector/models/role_router.py`; replace `app.py::_get_scope_agent`'s
  standalone primary/fallback construction with `app.py::_get_role_router`, built once from
  `INTELLIGENCE_REASONING_*`/`INTELLIGENCE_STANDARD_*` and shared by every agent; wire
  `AcceptanceVerifier(agent.router)` into `ResearchNodes` in `app.py::_get_scope_graph`. Recommended
  live defaults (documented in `.env.example`): GPT as the reasoning tier (Gemini has no free
  "xhigh"-equivalent — Pro is paid-only, Flash's ceiling is `thinking_level="high"`) and Gemini
  Flash (`reasoning_effort="low"`) as the standard tier, each with an optional GPT fallback.
  `SCOPE_MODEL`/`SCOPE_FALLBACK_MODEL` remain only for the opt-in live smoke tests. Completion means
  every agent's calls land on the correct tier by role name with no per-agent changes, covered
  offline in `tests/unit/test_scope_agent_wiring.py` and
  `tests/contract/test_role_model_routing.py` (FR-032a, FR-032b).
