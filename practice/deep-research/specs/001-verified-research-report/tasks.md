# Tasks: Verified Research Report

**Input**: Design documents from `/specs/001-verified-research-report/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Organization**: Tasks are ordered for an end-to-end vertical slice first, then grouped by the
five user stories. Every task includes an exact path and a completion condition with specification
references. Tests are included because the specification and request explicitly require them.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish the local Python application and testable package layout.

- [ ] T001 Create the package tree from `plan.md` under `src/intelligence_collector/` and `tests/`; completion means every planned module directory exists with package markers where needed (plan Project Structure).
- [ ] T002 Initialize Python 3.11 project dependencies and commands in `pyproject.toml`; completion means Streamlit, direct LangGraph, Pydantic, MCP client, vector-index adapter, Langfuse, Ragas, and pytest dependencies are declared without Deep Agents SDK (plan Technical Context; FR-030).
- [ ] T003 [P] Add repository configuration and safe local environment examples in `config.example.toml`, `.env.example`, and `.gitignore`; completion means secrets, VFS data, checkpoints, vector indexes, and permanent outputs are excluded from source control (constitution; FR-021).
- [ ] T004 [P] Add test fixtures and deterministic fake ports in `tests/conftest.py`, `tests/fixtures/fake_models.py`, `tests/fixtures/fake_mcp.py`, and `tests/fixtures/fake_vector.py`; completion means tests can run without network or paid model calls (quickstart).
- [ ] T005 [P] Add baseline lint/type/test commands in `pyproject.toml` and `tests/unit/test_imports.py`; completion means the package imports and the baseline test command runs successfully (plan Testing).

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build the contracts, persistence, safety, and graph foundation required by every user story.

**Checkpoint**: No user-story implementation starts until this phase passes its contract and persistence tests.

- [ ] T006 Define Pydantic domain contracts and enums in `src/intelligence_collector/contracts.py` for `ResearchGraphState`, `ScopedRequest`, `ResearchTodo`, `SourceRecord`, `ArtifactRef`, `IndexedChunk`, `ProgressAssessment`, `VerificationResult`, `ApprovalDecision`, and `PublicationRecord`; completion means required fields, stable IDs, schema versions, and enum values match `data-model.md` and contracts (FR-007, FR-016, FR-024).
- [ ] T007 [P] Implement path-safe atomic VFS repository in `src/intelligence_collector/storage/vfs.py`; completion means run-scoped writes reject traversal/cross-run paths, use atomic replacement, return content hashes, and never expose partial files (FR-010, FR-010b, FR-028a).
- [ ] T008 [P] Implement artifact repository and stage manifest in `src/intelligence_collector/storage/artifacts.py`; completion means Scope, Planning, Research, Synthesis, and Verification artifacts map to the VFS layout and preserve immutable versions and provenance (FR-010b, FR-021, SC-013).
- [ ] T009 [P] Implement checkpoint persistence in `src/intelligence_collector/storage/checkpoints.py`; completion means typed graph state saves/loads atomically by `thread_id`, detects invalid latest checkpoints, and selects the latest valid checkpoint or returns a review-required error (FR-021, FR-022).
- [ ] T010 [P] Implement local vector-index adapter in `src/intelligence_collector/storage/vector_index.py`; completion means raw source chunks index with source/artifact metadata and retrieval returns traceable chunk IDs, while index failure is explicit (FR-010a, FR-011, FR-012).
- [ ] T011 [P] Implement structured event/error and audit contracts in `src/intelligence_collector/observability/events.py`; completion means failures identify stage/task, retryability, artifact refs, and resume/review guidance (FR-029).
- [ ] T012 [P] Implement model router with role configuration, bounded retries, primary/fallback selection, and cost tier in `src/intelligence_collector/models/router.py`; completion means a forced primary failure selects the configured fallback exactly once within budget and emits a typed fallback event (plan model router; SC-011).
- [ ] T013 [P] Implement Langfuse tracing adapter in `src/intelligence_collector/observability/tracing.py`; completion means graph nodes, MCP calls, model selection/fallback, artifact writes, verification, interrupts, and publication can emit correlated run/thread spans without making success depend on tracing availability (plan observability).
- [ ] T014 [P] Implement input schema validation and prompt-injection/offensive/violent-content guardrail in `src/intelligence_collector/guardrails.py`; completion means rejected input returns a safe actionable `GuardrailDecision` and causes no graph, tool, checkpoint, VFS, or vector mutation (FR-002, User Story 2 scenario 4, plan guardrail).
- [ ] T015 Define direct LangGraph typed state, node wrapper, checkpoint configuration, and interrupt plumbing in `src/intelligence_collector/graph/state.py` and `src/intelligence_collector/graph/build.py`; completion means all transitions are graph edges, state is serializable, and approval interrupt can resume by thread ID (FR-021, FR-024, FR-030).
- [ ] T016 [P] Add contract tests for VFS, checkpoints, vector provenance, guardrail rejection, model fallback, and tracing in `tests/contract/test_storage_contracts.py`, `tests/contract/test_guardrail_contract.py`, `tests/contract/test_model_router_contract.py`, and `tests/contract/test_observability_contract.py`; completion means tests fail on path escape, invalid checkpoint, untraceable chunk, unsafe input, unbounded fallback, or missing correlation (constitution review gates).

## Phase 3: User Story 1 - Complete a Verified Research Run (Priority: P1) 🎯 MVP

**Goal**: Deliver one controlled-source, single-worker vertical slice from chat scope through verified report and approval-ready artifact.

**Independent Test**: Run `tests/integration/test_end_to_end_controlled.py` with fake MCP sources; assert scope, todos-before-retrieval, VFS evidence, indexing, ResearchComplete, synthesis, verification, UI state, and evidence-linked report (US1; SC-001–SC-005, SC-012).

### Tests for User Story 1

- [ ] T017 [P] [US1] Add controlled-source end-to-end test in `tests/integration/test_end_to_end_controlled.py`; completion means a bounded request reaches Awaiting Approval with no live network, and asserts `write_todos` precedes every MCP call (US1 scenarios 1–5; FR-005, FR-006, SC-001, SC-002).
- [ ] T018 [P] [US1] Add graph transition and state contract tests in `tests/contract/test_graph_transitions.py`; completion means invalid scope, failed todo write, incomplete research, and unverified publication cannot traverse their guarded edges (FR-003, FR-011, FR-024, FR-025).
- [ ] T019 [P] [US1] Add Streamlit AppTest for progress and report display in `tests/ui/test_chat_progress.py`; completion means current stage, task progress, verification outcome, evidence links, and report availability render from graph snapshots (FR-023, SC-004).

### Implementation for User Story 1

- [ ] T020 [US1] Implement Streamlit session/thread adapter in `src/intelligence_collector/app.py`; completion means chat turns map to a stable `thread_id`, UI reads graph snapshots, and UI code never writes authoritative artifacts (FR-001, FR-023).
- [ ] T021 [US1] Implement Scope Agent node and clarification state in `src/intelligence_collector/agents/scope.py`; completion means objective, boundaries, constraints, success criteria, and deliverable format are assessed for explicit/testable consistency, broad scope receives narrowing options, and clear scope writes `scope/brief_summary.v1.json` then edges automatically to Planning (FR-003–FR-005, US2 scenarios 1–5).
- [ ] T022 [US1] Implement `write_todos` tool in `src/intelligence_collector/tools/write_todos.py`; completion means it accepts only a non-empty list of `{todo_id, research_question, status=planned}`, persists `planning/todos.v1.json`, and blocks Research on any invalid or failed write (FR-006, FR-007, SC-002).
- [ ] T023 [US1] Implement Planner node in `src/intelligence_collector/agents/planner.py`; completion means Planning invokes `write_todos` before any retrieval and exposes actionable failure without mutating Research state (FR-006, FR-029).
- [ ] T024 [US1] Implement one isolated controlled-source MCP worker in `src/intelligence_collector/agents/worker.py` and MCP port in `src/intelligence_collector/tools/mcp.py`; completion means each task performs broad-to-narrow search, stops early or at its five-call hard limit, and writes raw capture, notes, excerpts, and evidence bundle VFS artifacts with provenance (FR-009–FR-010a, FR-009a–FR-009b, SC-011).
- [ ] T025 [US1] Implement Supervisor and `think_tool` in `src/intelligence_collector/agents/supervisor.py` and `src/intelligence_collector/tools/think_tool.py`; completion means task status/progress assessment is persisted and `ResearchComplete` is emitted only when every todo is completed, required VFS files exist, and vector indexing succeeds (FR-008, FR-011, SC-012).
- [ ] T026 [US1] Implement synthesis node in `src/intelligence_collector/agents/synthesis.py`; completion means indexed chunks/evidence are retrieved by provenance, previous and latest draft files are preserved, and no factual content is invented from chat history (FR-012, FR-010b).
- [ ] T027 [US1] Implement verifier node and pass result in `src/intelligence_collector/agents/verification.py`; completion means scope coverage and claim-to-evidence checks produce a typed result, and passing output exposes a versioned verified artifact with navigable evidence refs (FR-013–FR-015, SC-003).
- [ ] T028 [US1] Wire the MVP graph nodes and conditional edges in `src/intelligence_collector/graph/nodes.py` and `src/intelligence_collector/graph/routing.py`; completion means the controlled-source test executes Scope → Planning → Researching → ResearchComplete → Synthesis → Verification without a research-brief approval interrupt (FR-005, FR-024, SC-001).
- [ ] T029 [US1] Add MVP runbook and fixture references in `tests/integration/README.md`; completion means a developer can reproduce the controlled-source flow using the quickstart command and inspect every stage artifact (quickstart; SC-013).

**Checkpoint**: MVP vertical slice is independently demonstrable before parallel workers or live MCP are added.

## Phase 4: User Story 2 - Clarify an Ambiguous Request (Priority: P1)

**Goal**: Make scope clarification safe, multi-turn, bounded, and automatically connected to Planning.

**Independent Test**: `tests/integration/test_scope_clarification.py` submits broad, incomplete, contradictory, and unsafe requests; it asserts questions/narrowing, no retrieval before clarity, and automatic Planning after boundaries are explicit (US2; FR-002–FR-005).

- [ ] T030 [P] [US2] Add multi-turn scope tests in `tests/integration/test_scope_clarification.py`; completion means prior answers persist, only missing fields are requested, broad scopes receive narrowing suggestions, and no `write_todos`/MCP call occurs before scope completion (US2 scenarios 1–5).
- [ ] T031 [US2] Harden Scope Agent clarification schema and narrowing suggestions in `src/intelligence_collector/agents/scope.py`; completion means all five scope acceptance scenarios pass and each accepted scope writes one durable brief summary (FR-003, FR-004, FR-005).
- [ ] T032 [US2] Integrate guardrail and validation outcomes into Streamlit error rendering in `src/intelligence_collector/app.py` and `src/intelligence_collector/guardrails.py`; completion means unsafe/malformed input is actionable, safe, and mutation-free (FR-002, FR-029).

## Phase 5: User Story 3 - Recover from Failed Verification (Priority: P1)

**Goal**: Route typed BVF feedback to the smallest effective repair and terminate no-progress loops safely.

**Independent Test**: `tests/integration/test_bvf_routes.py` injects each issue type and repeated no-progress repairs; it asserts the correct route, preserved evidence, re-verification, bounded attempts, and `NEEDS_REVIEW` (US3; FR-016–FR-020, SC-005–SC-006).

- [ ] T033 [P] [US3] Add verifier issue and BVF routing tests in `tests/integration/test_bvf_routes.py`; completion means `DRAFT_DEFECT`, `EVIDENCE_GAP`, and `PLAN_GAP` each route correctly and unknown types are rejected (FR-016, FR-017).
- [ ] T034 [P] [US3] Add recovery-budget/no-progress tests in `tests/integration/test_recovery_limits.py`; completion means each repair re-verifies, preserves valid evidence, increments visible attempts, and exhaustion marks `NEEDS_REVIEW` (FR-018–FR-020, SC-006).
- [ ] T035 [US3] Implement typed verifier issues and deterministic BVF routing in `src/intelligence_collector/agents/verification.py` and `src/intelligence_collector/graph/routing.py`; completion means exactly one primary issue type selects draft revision, targeted research, or re-planning and a progress fingerprint blocks no-op loops (FR-016, FR-017).
- [ ] T036 [US3] Implement draft revision, targeted-gap research, and affected-task re-planning nodes in `src/intelligence_collector/agents/synthesis.py`, `src/intelligence_collector/agents/worker.py`, and `src/intelligence_collector/agents/planner.py`; completion means valid evidence is reused, changed artifacts invalidate prior verification, and every repaired candidate re-enters full verification (FR-017, FR-018).
- [ ] T037 [US3] Add `NEEDS_REVIEW` UI and artifact persistence in `src/intelligence_collector/app.py` and `src/intelligence_collector/storage/artifacts.py`; completion means unresolved feedback is visible and no UI/output labels the artifact verified (FR-020, FR-023).

## Phase 6: User Story 4 - Resume an Interrupted Run (Priority: P2)

**Goal**: Restore durable state and continue without repeating completed retrieval.

**Independent Test**: `tests/integration/test_resume.py` interrupts at each major stage, reloads the same thread, and asserts valid checkpoint restoration, intact artifacts, and no duplicate completed MCP calls (US4; FR-021–FR-022, SC-007).

- [ ] T038 [P] [US4] Add interruption/resume integration tests in `tests/integration/test_resume.py`; completion means Scope, Planning, Researching, Synthesis, and Verifying interruption points restore expected state and invalid checkpoints produce the documented recoverable error or `NEEDS_REVIEW` (US4 scenarios 1–3).
- [ ] T039 [US4] Integrate checkpoint save/load boundaries into every graph node in `src/intelligence_collector/graph/build.py` and `src/intelligence_collector/storage/checkpoints.py`; completion means state includes scope, todos, artifacts, assessments, drafts, verification history, counters, and publication state (FR-021).
- [ ] T040 [US4] Add idempotent task/artifact resume guards in `src/intelligence_collector/agents/worker.py` and `src/intelligence_collector/storage/artifacts.py`; completion means completed retrieval is skipped unless incomplete, invalidated, or explicitly repaired (FR-022, SC-007).

## Phase 7: User Story 5 - Control Publication (Priority: P2)

**Goal**: Bind human approval to an unchanged verified artifact and publish atomically.

**Independent Test**: `tests/integration/test_approval_binding.py` rejects, approves, mutates, and retries publication; it asserts no unauthorized permanent write and exact hash equality on success (US5; FR-025–FR-028a, SC-008–SC-009).

- [ ] T041 [P] [US5] Add approval/rejection and stale-hash tests in `tests/integration/test_approval_binding.py`; completion means unverified, failed, `NEEDS_REVIEW`, stale, rejected, and write-failure cases leave permanent output unchanged (US5 scenarios 1–5).
- [ ] T042 [US5] Implement approval interrupt/resume UI in `src/intelligence_collector/app.py` and `src/intelligence_collector/publication.py`; completion means only verified artifacts expose approval, decisions bind actor/time/hash, and rejection is durable without publication (FR-025–FR-027).
- [ ] T043 [US5] Implement atomic publisher in `src/intelligence_collector/publication.py`; completion means it is the only permanent-output writer, validates unchanged approved hash and verification ID, records success/failure, and never exposes partial output (FR-028, FR-028a, SC-008–SC-009).

## Phase 8: Cross-Cutting Resilience, Observability, and Acceptance

**Purpose**: Add explicit demonstrations and opt-in external validation after the controlled vertical slice works.

- [ ] T044 [P] Add model fallback integration test and trace assertions in `tests/integration/test_model_fallback.py`; completion means forced primary timeout selects fallback within budget and Langfuse records both attempts (plan model router; SC-011).
- [ ] T045 [P] Add Langfuse trace contract assertions in `tests/integration/test_langfuse_tracing.py`; completion means run, node, MCP, artifact, verification, interrupt, and publication spans share run/thread correlation, while tracing outage does not fabricate success (plan observability).
- [ ] T046 [P] Add repeatable Ragas faithfulness evaluation in `tests/evaluation/test_ragas_faithfulness.py` using controlled evidence contexts; completion means the fixed dataset produces a recorded score/configuration and evaluation artifacts remain separate from publication status (plan Ragas; SC-003).
- [ ] T047 [P] Add opt-in live MCP acceptance test in `tests/integration/test_live_mcp_opt_in.py`; completion means the test is skipped unless an explicit environment flag and configured MCP source are present, and when enabled it verifies provenance, five-call budgets, VFS artifacts, and vector indexing (FR-009–FR-011, SC-011–SC-013).
- [ ] T048 Run the complete quickstart validation in `specs/001-verified-research-report/quickstart.md`; completion means controlled end-to-end, ambiguity, resume, fallback, all BVF routes, guardrail, faithfulness, and approval-binding demonstrations pass before any optimization work (all acceptance criteria).
- [ ] T049 [P] Add operational documentation for VFS layout, checkpoint recovery, model configuration, Langfuse opt-in, Ragas evaluation, and live MCP opt-in in `docs/deep-research-runbook.md`; completion means a new developer can reproduce controlled tests without credentials and understands permanent-output safeguards (constitution; plan contracts).
- [ ] T050 Review and harden security/cost boundaries in `src/intelligence_collector/config.py`, `src/intelligence_collector/guardrails.py`, `src/intelligence_collector/storage/vfs.py`, and `src/intelligence_collector/models/router.py`; completion means secrets never enter artifacts/logs, path escapes fail, search/recovery/model budgets are finite, and no Deep Agents SDK is present (constitution; FR-002, FR-019, FR-028a).

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
