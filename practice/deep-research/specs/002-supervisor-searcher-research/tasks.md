---

description: "Task list for the supervisor–searcher research workflow"
---

# Tasks: Supervisor–Searcher Research Workflow

**Input**: Design documents from `specs/002-supervisor-searcher-research/`

**Prerequisites**: [plan.md](plan.md), [spec.md](spec.md), [research.md](research.md),
[data-model.md](data-model.md), [contracts/](contracts/), [quickstart.md](quickstart.md)

**Tests**: Included. The constitution (Delivery and Review Gates) requires automated tests for
rejected input, legal/illegal transitions, retry exhaustion, resume, persistence, evidence links
and verification failure. All tests run offline (`ControlledMCP`, `LocalVectorIndex`, fake
routers).

**Existing behavior relied on**: FR-025 (acceptance criteria verdicts) and the per-criterion part of
FR-027 are met by the existing coverage gate (`agents/report_judge.py`); T048 guards them with a
regression check.

**Organization**: Tasks are grouped by user story (spec.md). Paths are relative to the project
root `practice/deep-research/`; package code is under `src/intelligence_collector/`.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: US1–US5 from spec.md

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Baseline, configuration and model roles every story uses

- [X] T001 Run `uv run pytest -q` and `uv run ruff check src tests` from the project root and confirm both are green before any change; if not, stop and report the failures
- [X] T002 [P] Add the new model roles to `src/intelligence_collector/models/token_limits.py` with these output caps: `result_evaluator` 1024, `query_rewriter` 512, `searcher_report` 1536, `claim_judge` 3072; keep `research_compressor` until T037 removes its last use
- [X] T003 [P] Map the new roles to tiers in `src/intelligence_collector/models/factory.py`: `result_evaluator`, `query_rewriter`, `searcher_report` → standard tier; `claim_judge` → reasoning tier (gap judgment reuses the existing `supervisor` role)
- [X] T004 [P] Add settings to `src/intelligence_collector/config.py` (`Settings.from_env`, using `_bounded_int`): `max_sources_per_task` default 5 (bound stays ≤ 10), `lookup_top_k` from `INTELLIGENCE_LOOKUP_TOP_K` default 5 range 1–20, `max_follow_ups_per_run` from `INTELLIGENCE_MAX_FOLLOW_UPS_PER_RUN` default 4 range 0–8, `max_follow_ups_per_answer` from `INTELLIGENCE_MAX_FOLLOW_UPS_PER_ANSWER` default 2 range 0–4, `report_max_words` from `INTELLIGENCE_REPORT_MAX_WORDS` default 400 range 100–1000; document each in `.env.example`
- [X] T005 Create the benchmark fixture folder `evals/supervisor_searcher/` with `README.md` and three JSON fixtures: `results.json` (search rows labeled relevant / off_topic / duplicate / no_usable_content per question), `gaps.json` (plans with planted essential and non-essential gaps and the page that closes each), `claims.json` (draft sentences labeled supported / mis-cited / partly supported / unsupported / uncited with their evidence snapshot); then create `scripts/benchmark_supervisor_searcher.py` (live, opt-in: runs only with `--run-live`, the repo convention for paid scripts, and writes `{"live": false}` when the model or Tavily settings are missing) over the questions in `evals/supervisor_searcher/questions.json` that runs each fully scoped benchmark question through the current workflow end to end: a clarification pause is recorded as `skipped`, plan approval is resumed with `{"approved": true}`, the run stops at the publication-approval pause after verification and never publishes; errors are stored only through `redact_error`; it writes `evals/supervisor_searcher/baseline.json` with the share of drafts needing at least one repair cycle (SC-007 baseline) and the mean searches per task (SC-008 baseline); run it now, **before T002–T004 or any other code change**, and commit `baseline.json` (if keys are unavailable, record `{"live": false}` and note in the README that SC-007/SC-008 cannot be compared)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Contracts, evidence points in the plan, state, passage lookup and quote checking — used by several stories

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T006 Add enums to `src/intelligence_collector/contracts.py` as `str, Enum` with exactly these values: `PointStatus` (`answered`, `partial`, `missing`), `RejectReason` (`off_topic`, `no_usable_content`, `duplicate`, `low_credibility`, `outside_scope`), `StopReason` (`all_points_covered`, `search_budget_spent`, `page_budget_full`, `no_progress`, `search_failures`, `model_finished`), `ReportStatus` (`complete`, `corrected`, `unreadable`), `FailureKind` (`technical`, `content`), `TaskOrigin` (`plan`, `follow_up`), `ClaimVerdict` (`supported`, `partly_supported`, `unsupported`, `contradicted`)
- [X] T007 Add `EvidencePoint` to `src/intelligence_collector/contracts.py` (`point_id` matching `E1`…`E4`, `text` "non-blank, ≤ 300 chars", `required_answers: list[str]`, property `essential = bool(required_answers)`) and extend `ResearchTodo` with `evidence_points: list[EvidencePoint]` ("1–4"; empty allowed only for legacy todos handled in T013), `origin: TaskOrigin = plan`, `round: int = 1` ("≥ 1"), `gap_point_ids: list[str] = []`; reject duplicate `point_id`s in one todo
- [X] T008 Extend `TodoPlan` in `src/intelligence_collector/contracts.py` with `evidence_points: list[list[EvidencePointDraft]]` where `EvidencePointDraft = {text: str, required_answers: list[int]}`; validate "same length as `questions`; 1–4 per task; indexes 1-based" (index range is checked against the brief in T011)
- [X] T009 Add the remaining contracts to `src/intelligence_collector/contracts.py` per [data-model.md](data-model.md): `ResultEvaluation` (`reason` "required when rejected, absent when accepted"; `covers` only for accepted; `snippet_only: bool` "accepted with only the search snippet"), `PointReport` (`support` "required non-empty for answered/partial"), `Finding`, `SearcherReportDraft`, `SearcherReport` (`findings` "≤ 5"; `report_status` "`corrected` when `corrections` is non-empty; `unreadable` when no part of the model output could be read"), `PassageLookupRecord` (`results` "≤ 5", `fallback` in `none`/`vector_only`/`combined_query_only`), `Gap`, `GapAnalysis` (`ended` in `no_essential_gaps`/`caps_reached`/`stalled`/`turn_limit`/`follow_ups_created`; `turns_used` "≤ 4"; `turn_limit_reached: bool` "true whenever the 4 turns were spent, whether or not follow-ups were created"; `stall_count` "consecutive follow-up rounds (round ≥ 2) with `closed_essential == 0`; round 1 never counts"), `Claim`, `ClaimEvaluation` (`supported_share` "1.0 when none")
- [X] T010 [P] Write contract validation tests for T006–T009 in `tests/unit/test_supervisor_searcher_contracts.py` (blank point text, > 300 chars, 5 points, duplicate ids, rejected without reason, accepted with reason, answered without support, 6 findings, mismatched `evidence_points` length)
- [X] T011 Update the planner for evidence points: extend `WRITE_TODOS_TOOL` in `src/intelligence_collector/agents/tools.py` with `evidence_points`; in `src/intelligence_collector/agents/planner.py` validate indexes against `brief.checklist` and that every checklist item is linked by at least one point (on failure answer with a `ToolMessage` naming the unlinked items and retry within `max_attempts`); make `create_todos` build `EvidencePoint`s (`E1`… per task, `required_answers` as checklist texts) and keep `evidence_needed` as the points joined with "; "; update the planner prompt in `src/intelligence_collector/agents/prompts.py` to require 1–4 linked evidence points per task
- [X] T012 Update `plan_record` in `src/intelligence_collector/agents/planner.py` so each hashed task includes `evidence_points` (`point_id`, `text`, `required_answers`) and the record carries an unhashed `follow_ups: []` list; extend `tests/unit/test_planner_write_todos_tool.py` for: unlinked required answer → retry, 5 points → retry, hash changes when a point changes, hash unchanged when `follow_ups` changes
- [X] T013 Add state and migration: in `src/intelligence_collector/graph/state.py` add `research_round: int`, `gap_stall_count: int`, `follow_up_count: int`, `gap_analysis: dict | None`, `claim_support_score: float | None`, `claim_evaluation_ref: str`; in `src/intelligence_collector/graph/migration.py` default them (1, 0, 0, None, None, "") and give each legacy todo without evidence points one derived point (`E1`, text = `evidence_needed` or the question, linked to every checklist item of `scoped_request`) without touching the stored plan or its hash
- [X] T014 Add `claim_support_score: None` and `claim_evaluation_ref: ""` to `invalidate_draft_authority()` in `src/intelligence_collector/graph/nodes/common.py`, and reset `research_round=1`, `gap_stall_count=0`, `follow_up_count=0`, `gap_analysis=None` wherever a new plan or scope starts: `PlannerNodes.planning` (`graph/nodes/scope.py`), the `scope` route of `ApprovalNodes.classify_feedback` (`graph/nodes/review.py`) and the `rescope` action of `RecoveryNodes.apply` (`graph/nodes/recovery.py`); extend `tests/unit/test_draft_authority.py` and add a migration test in `tests/unit/test_checkpoint_migration_evidence_points.py`
- [X] T015 [P] Add `documents(run_id) -> list[IndexedChunk]` to the `VectorIndex` protocol and `LocalVectorIndex` in `src/intelligence_collector/storage/vector_index.py`, and to `ChromaVectorIndex` in `src/intelligence_collector/storage/chroma_vector_index.py` via `collection.get(where={"run_id": run_id}, include=["documents", "metadatas"])`; also store `chunk_index` in Chroma upsert metadata (parsed from the chunk id or `IndexedChunk.metadata`)
- [X] T016 Implement `PassageLookup` in new `src/intelligence_collector/storage/passage_lookup.py` per [contracts/passage-lookup.md](contracts/passage-lookup.md): per query vector retrieval (`limit=candidates`, default 10) and in-house Okapi BM25 (lowercase `\w+`, k1 = 1.5, b = 0.75) over `documents(run_id)`; filter by `allowed_source_ids`; Reciprocal Rank Fusion with `rrf_k = 60`; return `top_k` (default from `lookup_top_k`) chunks, ties by vector rank then chunk id; fallback `vector_only` when `documents()` fails; redacted `error`; never writes the index (depends on T015)
- [X] T017 [P] Write `tests/unit/test_passage_lookup.py`: a passage containing "281 TB" reaches the top 5 through the keyword ranking while vector ranking misses it; allowed-source filter; fusion order; fallback recorded; empty run returns no chunks and no error
- [X] T018 [P] Add quote helpers to `src/intelligence_collector/source_verification.py`: `normalize_for_quote(text)` (NFKC, casefold, collapsed whitespace, unified quotes and dashes) and `quote_in_text(quote, text) -> bool`; add `stored_text_for_source(vfs, run_id, source) -> str` in `src/intelligence_collector/storage/evidence_index.py` reusing `stored_source_text`; test both in `tests/unit/test_quote_check.py`

**Checkpoint**: Foundation ready — plans carry evidence points, state is migrated, lookup and quote checks work

---

## Phase 3: User Story 1 - Searchers collect only evidence worth keeping (Priority: P1) 🎯 MVP

**Goal**: Each searcher evaluates every result, steers its queries to uncovered evidence points, stops for a recorded reason, and persists only accepted pages (every chunk indexed) before reporting.

**Independent Test**: One task against `ControlledMCP` returning relevant, off-topic, duplicate and empty rows: only relevant rows are accepted and stored, each rejection has a reason, the second query targets missing points, the searcher stops within limits with a recorded reason, and every accepted page is searchable with provenance.

### Tests for User Story 1

- [X] T019 [P] [US1] Write `tests/unit/test_result_evaluation.py`: duplicate and empty rows rejected without a model call (`duplicate`, `no_usable_content`); model rejects with `off_topic`/`low_credibility`/`outside_scope`; a row index the model omits becomes `rejected` (`off_topic`) with a recorded correction; `covers` keeps only known point ids
- [X] T020 [P] [US1] Write `tests/unit/test_researcher_stop.py`: each `StopReason` (`all_points_covered`, `search_budget_spent` at 5 searches, `page_budget_full` at 5 accepted pages, `no_progress` after 2 consecutive searches with no accepted row, `search_failures` after 2 consecutive failures, where a failed result evaluation also counts as a failed search, `model_finished`); a repeated query (same normalized words) is refused and does not spend the budget
- [X] T021 [P] [US1] Extend `tests/integration/test_worker_mcp.py`: rejected rows are never captured, claimed or indexed; accepted pages are captured once and every chunk of the cleaned page is indexed; `evaluations.json` lists every decision; a page accepted by two parallel tasks is stored once

### Implementation for User Story 1

- [X] T022 [P] [US1] Add the result-evaluation prompt (task question, evidence points `E1…`, rows with title/URL/selected passages; credibility guidance: primary or official sources, recency for time-sensitive points, consistency with other sources) to `src/intelligence_collector/agents/prompts.py`, and its structured-output model (`rows: [{index, decision, reason?, covers}]`) to `src/intelligence_collector/contracts.py`
- [X] T023 [US1] Implement `evaluate_results(rows, todo, seen_urls, router, ...) -> list[ResultEvaluation]` in new `src/intelligence_collector/agents/result_evaluation.py`: deterministic `duplicate` / `no_usable_content` first, one `result_evaluator` call for the rest, validation rules from [contracts/supervisor-tools.md](contracts/supervisor-tools.md), redacted errors; when the evaluation call fails, accept no row of that search and mark the search as failed so it counts toward `search_failures` (FR-005); set `snippet_only = true` for an accepted row without a readable page (depends on T022)
- [X] T024 [US1] Rework the search loop in `src/intelligence_collector/agents/researcher.py`: after each search read pages, call `evaluate_results`, claim (`UrlClaims.claim`) only accepted rows, count only accepted rows toward the page budget, track covered evidence point ids, refuse repeated queries (normalized: casefold, collapsed whitespace, sorted words), include uncovered points and accepted/rejected lines in the tool reply, and stop in `route_after_llm` with a `StopReason`; add `evaluations`, `queries`, `stop_reason`, `covered_points`, `failure_kind` to `ResearchOutcome`; default `max_sources` to the new setting (5)
- [X] T025 [US1] Split `ResearchAgent.research` into `search(...) -> ResearchOutcome` (the loop without compression) and `summarize(outcome, ...)` (the existing compression, kept only until T037 replaces it) in `src/intelligence_collector/agents/researcher.py`; update `RESEARCHER_TEMPLATE` in `src/intelligence_collector/agents/prompts.py` to show the task's evidence points and to target uncovered ones
- [X] T026 [US1] Split persistence in `src/intelligence_collector/agents/worker.py` into `persist_pages(...)` (accepted rows only: raw captures with a `snippet_only` flag, `UrlRegistry`, index every chunk, `evaluations.json` at `<prefix>/evaluations.json`) and `write_evidence_bundle(...)` (written last, so `already_researched` still means done); keep `persist_research` as a thin wrapper for existing callers
- [X] T027 [US1] Update `ResearchSupervisor._gather_sub_agents` / `_research_one` in `src/intelligence_collector/agents/supervisor.py`: per job run `search` in a worker thread, write `retrieved_outcome.json` (now including evaluations, queries, stop reason), call `persist_pages` on the event-loop thread, then run `summarize` in a worker thread and `write_evidence_bundle`, all inside one concurrency slot; pass the todo's evidence points in `_sub_agent_question`
- [X] T028 [US1] Add `traced` spans `result_evaluation` and keep `mcp.search` in `src/intelligence_collector/agents/researcher.py`; ensure every new error string goes through `redact_error`

**Checkpoint**: US1 works — searchers keep only good evidence and stop for recorded reasons (reports still use the interim summary)

---

## Phase 4: User Story 2 - Each searcher returns a short, structured report (Priority: P1)

**Goal**: The searcher closes its own gaps with a hybrid lookup over the run's stored pages, then returns a ≤ 400-word report whose answered points carry verified verbatim quotes; invalid parts become `missing`.

**Independent Test**: Complete a task whose answer to one evidence point sits in a chunk the searcher never read: the point is reported answered with a verified quote and no extra web search; every point has a status; the report is within limits and readable after resume.

### Tests for User Story 2

- [X] T029 [P] [US2] Write `tests/unit/test_searcher_report.py`: self-lookup answers a deep-chunk point without spending a search; quote not found → point `missing` with a correction; unknown or rejected source dropped with the finding citing it; absent point added as `missing`; quote > 30 words cut to a verifying prefix; findings dropped until ≤ 400 words while points stay; unreadable output → all `missing` and `report_status = unreadable`; any correction → `report_status = corrected`, none → `complete`; phrasing failure → `combined_query_only`
- [X] T030 [P] [US2] Extend `tests/contract/test_structured_output_schemas.py` with `SearcherReportDraft` and the query-rewrite output schema, and check the stored `report.json` shape against [contracts/searcher-report.md](contracts/searcher-report.md)

### Implementation for User Story 2

- [X] T031 [P] [US2] Add the query-rewrite prompt (2–3 phrasings per point, "at least one in the language of the listed source titles") and the searcher-report prompt (per point: status and `{source_id, quote}` support; ≤ 5 findings with source ids; quotes copied verbatim, ≤ 30 words) to `src/intelligence_collector/agents/prompts.py`
- [X] T032 [US2] Implement self-lookup in new `src/intelligence_collector/agents/searcher_report.py`: for every evidence point not covered by passages read while searching (essential or not), build the combined query (`point text + " — " + task question`) plus phrasings from one batched `query_rewriter` call, run `PassageLookup` restricted to the sources of the run's active tasks stored so far (its own accepted sources and those of finished active tasks; pages of a superseded plan are excluded), and write `<prefix>/lookups.json`; lookups never count against the search budget and never wait for running tasks (depends on T016, T031)
- [X] T033 [US2] Implement report drafting and validation in `src/intelligence_collector/agents/searcher_report.py`: one `searcher_report` call over the passages read plus lookup chunks, then code rules from [contracts/searcher-report.md](contracts/searcher-report.md) (quote check with `quote_in_text` against `stored_text_for_source`, accepted-source check, one entry per point, "≤ 30 words per quote", "≤ 5 findings", "≤ 400 words", corrections list), counts/queries/stop reason/failure kind filled from the outcome, `report_status` set to `complete`, `corrected` or `unreadable`; write `<prefix>/report.json` (depends on T018, T032)
- [X] T034 [US2] Add `ResearchAgent.report(outcome, todo, ...) -> SearcherReport` in `src/intelligence_collector/agents/researcher.py` delegating to `searcher_report.py`; on a searcher that found no accepted result, report every point `missing` with the stop reason and `failure_kind` (`technical` when searches failed, `content` otherwise)
- [X] T035 [US2] Derive the evidence bundle from the report in `write_evidence_bundle` (`src/intelligence_collector/agents/worker.py`): `summary` = findings, `sufficient` = every point answered, `unresolved_gaps` = missing/partial point texts, each source's `excerpt` = its selected passages plus the looked-up chunks that support a verified quote (deduplicated)
- [X] T036 [US2] In `src/intelligence_collector/agents/supervisor.py` replace `summarize` with `report`, render `ConductResearch` replies from `report.json` (header, one line per point with status, source and quote, findings, counts, stop reason), and make `_result_summary` read the latest attempt's `report.json`; in `_decide`, refuse `ConductResearch` for a task that already has a report unless that report's `failure_kind` is `technical` (`Refused: content gaps are closed by follow-up tasks.`, FR-002a); resume reuses an existing `report.json`
- [X] T037 [US2] Remove the interim compression: delete `summarize`, `COMPRESS_TEMPLATE` usage and the `research_compressor` role from `src/intelligence_collector/agents/researcher.py`, `src/intelligence_collector/agents/prompts.py` and `src/intelligence_collector/models/token_limits.py`; keep `ResearchFindings` only for reading legacy `retrieved_outcome.json` files

**Checkpoint**: US1 + US2 work — every task ends with a verified short report

---

## Phase 5: User Story 3 - The supervisor closes essential gaps with targeted follow-up tasks (Priority: P1)

**Goal**: After every round, `analyze_gaps` classifies gaps from the plan links, looks up essential gaps across the whole run, and lets the supervisor create bounded follow-up tasks for the next round; research ends on no gaps, caps or two stalled rounds.

**Independent Test**: A plan where one task misses an essential point and another misses a non-essential point: exactly one follow-up is created after the whole round, it targets the essential point, the non-essential gap becomes a limitation, caps are never exceeded, and a gap answered on another task's page creates no follow-up.

### Tests for User Story 3

- [X] T038 [P] [US3] Write `tests/unit/test_gap_analysis.py`: classification uses only plan links; run-wide lookup marks a gap covered only with a verified quote; `CreateFollowUp` refusals for unknown or covered gap keys, links outside 1..N, "≤ 2 follow-ups per required answer, ≤ 4 per run", and duplicate questions; the stall counter ignores round 1 and increments only for follow-up rounds (round ≥ 2) that close no essential gap; after 4 supervisor turns the analysis ends with the follow-ups accepted so far (`ended = turn_limit` when none); the analysis reads the latest attempt's report of each task
- [X] T039 [P] [US3] Write `tests/integration/test_supervisor_rounds.py` with `ControlledMCP` and a fake supervisor router: follow-ups are created only after every task of round 1 finished (tasks delegated in two batches); a follow-up runs as round 2 and closes the gap; a gap covered by another task's page creates no follow-up; caps stop follow-ups; two rounds without progress end research; an out-of-scope gap leads to `ProposeReplan`; MCP failure path: `ControlledMCP` raising on every search for one task → the searcher stops with `search_failures`, its report has `failure_kind = technical`, a second attempt is allowed, and gap analysis treats its essential points as gaps
- [X] T040 [P] [US3] Extend `tests/contract/test_graph_transitions.py` for `route_after_delegation` (replan → `replan`, otherwise → `analyze_gaps`) and `route_after_gap_analysis` (follow-ups created → `researching`, otherwise → `research_complete`), and check that the plan-approval guard in `researching` still passes after follow-ups are added

### Implementation for User Story 3

- [X] T041 [P] [US3] Add the `CreateFollowUp` tool schema (`question`, `evidence_points: [{text, required_answers: [int]}]`, `gap_point_keys`, `reason`) to `src/intelligence_collector/agents/tools.py`, and the gap coverage-judgment and follow-up-writing prompts to `src/intelligence_collector/agents/prompts.py`
- [X] T042 [US3] Implement `src/intelligence_collector/agents/gap_analysis.py`: collect missing/partial points from the reports of the active tasks; classify essential when `required_answers` is non-empty; run `PassageLookup` over the active catalog sources for every essential gap; one `supervisor` structured call judges coverage, quotes verified with `quote_in_text`; offer `CreateFollowUp` (and `ProposeReplan`) in a tool loop of at most 4 supervisor turns per analysis (FR-019a) with the refusals from [contracts/supervisor-tools.md](contracts/supervisor-tools.md); count stalls only for follow-up rounds (round ≥ 2); read the latest attempt's report of each task; restrict lookups to the sources of the run's active tasks; non-essential gaps → limitations; return a `GapAnalysis` and write `research/rounds/round-{n}.json` and `round-{n}.lookups.json`; reuse an existing round file on resume (depends on T016, T041)
- [X] T043 [US3] Add `Node.ANALYZE_GAPS = "analyze_gaps"` (agent "Research Supervisor") to `src/intelligence_collector/graph/nodes/common.py` and implement `ResearchNodes.analyze_gaps` in `src/intelligence_collector/graph/nodes/research.py`: append follow-up todos (`origin=follow_up`, `round=research_round + 1`, `gap_point_ids`, evidence points `E1…`) to `todos` and `plan["follow_ups"]`, update `research_round`, `follow_up_count`, `gap_stall_count`, `gap_analysis`, activity and current action; return only changed keys
- [X] T044 [US3] Wire routing: in `src/intelligence_collector/graph/routing.py` change `route_after_delegation` to return `Node.ANALYZE_GAPS` when there is no replan and add `route_after_gap_analysis`; in `src/intelligence_collector/graph/build.py` add the node and edges `researching → analyze_gaps → researching | research_complete`
- [X] T045 [US3] Compute readiness from the latest `GapAnalysis` in `ResearchNodes.research_complete` (`src/intelligence_collector/graph/nodes/research.py`): READY when no essential gap remains; PARTIAL when only non-essential or capped/stalled essential gaps remain with usable evidence; INSUFFICIENT when no usable evidence exists; route to `prepare_gap_research` only for INSUFFICIENT
- [X] T046 [US3] Update `ResearchSupervisor._task_list` in `src/intelligence_collector/agents/supervisor.py` to show the round number and each task's evidence points, and log `gap_analysis`, `follow_up` and `follow_up_refused` entries through `DecisionLog`; the supervisor turn limit (8) applies per round

**Checkpoint**: US1–US3 work — research runs in bounded rounds with follow-ups for essential gaps only

---

## Phase 6: User Story 4 - The final report is evaluated claim by claim (Priority: P1)

**Goal**: Each fact-stating sentence is judged only against the passages the writer was shown; every cited source must support part of it and together all of it; this is the faithfulness gate. RAGAS Faithfulness becomes informational.

**Independent Test**: A draft with a supported sentence, a mis-cited sentence, a partly supported sentence, an uncited factual sentence and an unmet required answer: only the first passes, each failure names the sentence and source, and the draft does not pass.

### Tests for User Story 4

- [X] T047 [P] [US4] Write `tests/unit/test_claim_judge.py`: sentence segmentation (list items, tables, citations with commas, `Assessment:` and gap sentences excluded); rules `uncited`, `unknown_source`, `source_supports_nothing` (naming the source), `not_fully_supported`, `contradicted`; a sentence combining two sources that each support one part passes; `supported_share`; a missing claim id gets one supplemental call, then ERROR
- [X] T048 [P] [US4] Extend `tests/integration/test_bvf_loop.py`: claim failures route to the `vfs` repair; the judge reads only `synthesis/evidence.v{N}.json`; a presentation repair copies the previous snapshot; RAGAS faithfulness appears only in `evaluating_quality` and a low or failed RAGAS score never blocks; regression: per-criterion coverage verdicts (FR-025) are still produced by the existing coverage gate and stored in the verification record with the draft version (FR-027)

### Implementation for User Story 4

- [X] T049 [US4] Write the per-draft evidence snapshot in `BVFNodes.synthesizing` (`src/intelligence_collector/graph/nodes/report.py`): `synthesis/evidence.v{N}.json` with, for every source id shown to the writer, `title`, `url` and the exact text shown (excerpt truncated by `EXCERPT_CHARS`) and the questions it was shown under; a presentation-only repair copies the previous snapshot with `copied_from_version`
- [X] T050 [P] [US4] Add the claim-judge prompt (per sentence: `claim_id`, sentence, each cited source's snapshot text; output `states_fact`, `verdict`, `per_source[{source_id, supports_part}]`, `unsupported_parts`) to `src/intelligence_collector/agents/prompts.py`
- [X] T051 [US4] Implement `src/intelligence_collector/agents/claim_judge.py` per [contracts/claim-evaluation.md](contracts/claim-evaluation.md): segmentation over `report_sections`, batches "≤ 10 sentences, ≤ 4 calls in flight", one supplemental call for missing ids, rules and failure codes in code, `ClaimEvaluation` and a `VerificationCheck(name="claims", gate="faithfulness", missing_points ≤ 8 sentences)` (depends on T050)
- [X] T052 [US4] Replace the RAGAS gate in `BVFNodes.verifying` (`src/intelligence_collector/graph/nodes/report.py`) with the claim judge reading the snapshot; write `verification/claims.{verification_id}.json`; set `claim_support_score` and `claim_evaluation_ref`; a judge failure is ERROR through the existing `_error` path with saved coverage checks
- [X] T053 [US4] Move RAGAS Faithfulness into `BVFNodes.evaluating_quality` (`src/intelligence_collector/graph/nodes/report.py`) using the snapshot texts of cited sources as contexts; record `faithfulness_score`/`faithfulness_error` in the quality record; never re-enter the loop; update `SCORE_FIELDS` in `src/intelligence_collector/graph/instrumentation.py` (faithfulness producer `evaluating_quality`, new `claim_support` from `verifying`)
- [X] T054 [US4] Wire the claim judge into `ResearchNodes`/`BVFNodes` constructors (`src/intelligence_collector/graph/nodes/research.py`, `src/intelligence_collector/graph/nodes/report.py`) and the app runtime (`src/intelligence_collector/runtime.py`, `app.py`) with the `claim_judge` router; RAGAS stays optional
- [X] T055 [US4] Update the UI gate card in `src/intelligence_collector/ui/html.py` (`verification_gates`: faithfulness gate shows claim results and `claim_support_score`) and the quality panel in `src/intelligence_collector/ui/components/panel.py` (RAGAS faithfulness listed under quality signals with the existing threshold warning)

**Checkpoint**: US1–US4 work — the report passes only when every factual sentence is supported by its own cited passages

---

## Phase 7: User Story 5 - The research process is observable and resumable (Priority: P2)

**Goal**: Per task, users see queries, accepted/rejected results, stop reason, report status and follow-ups; per draft, claim verdicts. An interrupted run resumes without repeating completed work.

**Independent Test**: Interrupt a run after some tasks finished, resume it: finished tasks are not searched again, round and follow-up decisions are preserved, and the activity view shows the full history.

### Tests for User Story 5

- [X] T056 [P] [US5] Add a resume test to `tests/integration/test_supervisor_rounds.py` (`-k resume`): interrupt after round 1's analysis and after half of round 2; on resume no search, capture, report or round analysis is repeated and `research_round`, `follow_up_count` and `gap_stall_count` are preserved
- [X] T057 [P] [US5] Write `tests/ui/test_activity_supervisor_searcher.py`: the activity renders per-task queries, accepted/rejected counts with reasons, stop reason, report status, follow-up tasks with their gaps, and per-draft claim verdicts, with all text escaped

### Implementation for User Story 5

- [X] T058 [US5] Render the new records in the UI: per-task research details from `evaluations.json` and `report.json`, round summaries from `research/rounds/round-{n}.json`, follow-up tasks with their gaps, and claim verdicts from `claim_evaluation_ref`, where each claim expands to the exact snapshot text of every cited source (from `synthesis/evidence.v{N}.json`) so a reviewer reaches the passage in one click (SC-006), and snippet-only sources carry a "snippet only" label (FR-012), in `src/intelligence_collector/ui/components/panel.py`, `src/intelligence_collector/ui/html.py` and `src/intelligence_collector/ui/evidence.py`
- [X] T059 [US5] Make every new step idempotent on resume: `_research_one` reuses `retrieved_outcome.json` (with evaluations) and `report.json`; `persist_pages` skips already-registered URLs; `analyze_gaps` reuses an existing round file; files in `src/intelligence_collector/agents/supervisor.py`, `src/intelligence_collector/agents/worker.py`, `src/intelligence_collector/agents/gap_analysis.py`
- [X] T060 [US5] Add `traced` spans `passage.lookup`, `searcher.report`, `gap_analysis` and `claim_judge` (low-cardinality metadata only: counts, ids, fallback) in `src/intelligence_collector/storage/passage_lookup.py`, `src/intelligence_collector/agents/searcher_report.py`, `src/intelligence_collector/agents/gap_analysis.py`, `src/intelligence_collector/agents/claim_judge.py`

**Checkpoint**: All user stories work and are observable

---

## Phase 8: Polish & Cross-Cutting Concerns

- [X] T061 [P] Extend `tests/unit/test_redaction.py`: a provider error containing a key in an evaluation, lookup, report, gap analysis or claim judge failure is stored and logged without the key; the benchmark runner's `baseline.json` and `benchmark_results.json` never contain a credential
- [X] T062 [P] Benchmark in two parts: (a) offline `tests/evaluation/test_supervisor_searcher_benchmark.py` that checks the metric computations on the `evals/supervisor_searcher/` fixtures with scripted routers (precision/recall of result evaluation, gap closure counting, claim-error detection, repair-cycle share, searches per task); (b) extend `scripts/benchmark_supervisor_searcher.py` (same `--run-live` rule as T005) to run the new workflow live and write `evals/supervisor_searcher/benchmark_results.json` with result-evaluation precision/recall (SC-002), planted-gap closure and no-gap follow-ups (SC-004), claim-error detection and false flags (SC-005), repair-cycle share compared with `baseline.json` (SC-007) and searches per task compared with `baseline.json` (SC-008)
- [X] T063 [P] Update `docs/research-workflow.md` and `docs/observability.md` for evidence points, result evaluation, stop reasons, searcher reports, rounds and follow-ups, passage lookup, the claim gate and the new trace spans
- [X] T064 Sync `specs/002-supervisor-searcher-research/spec.md`, `plan.md` and this `tasks.md` with the implemented behavior (mark completed tasks, record any deviation)
- [X] T065 Run every scenario in [quickstart.md](quickstart.md), then `uv run pytest -q` and `uv run ruff check src tests`; fix any failure before closing the feature

### Implementation deviations

- T034: no `ResearchAgent.report`; `SearchTask` (`src/intelligence_collector/agents/search_task.py`) runs search → persist → lookup → report → bundle as steps the supervisor schedules (see plan.md, Implementation Notes). The "no accepted result" report is `searcher_report.empty_report`.
- T022–T024: result evaluation is its own module, `src/intelligence_collector/agents/result_evaluation.py`.
- T042: gap closure checks later follow-up reports (`covered_by_report`) before the lookup.
- T051: a sentence with a citation is always a factual claim (spec FR-022 updated).
- T053/T054: `research_compressor` retired; its env overrides are ignored via `RETIRED_ROLES`.
- T059: the searcher report is saved right after drafting, not only in the final step.
- T062: metrics live in `src/intelligence_collector/observability/supervisor_benchmark.py`; `benchmark_results.json` is written only by a `--run-live` run, which has not been run yet.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: none; T005 must run right after T001 and before T002–T004 (its baseline measures the unchanged code)
- **Foundational (Phase 2)**: after Setup; blocks every story
- **US1 (Phase 3)**: after Foundational
- **US2 (Phase 4)**: after US1 (needs the search/persist split, T025–T027)
- **US3 (Phase 5)**: after US2 (gap analysis reads searcher reports)
- **US4 (Phase 6)**: after Foundational only; independent of US1–US3
- **US5 (Phase 7)**: after US3 and US4
- **Polish (Phase 8)**: after the stories you ship

### Within Each Story

- Tests first; they must fail before the implementation tasks.
- Prompts/contracts before agent modules; agent modules before graph wiring; wiring before UI.

### Key task dependencies

- T016 → T032, T042 (lookup)
- T018 → T033, T042 (quote check)
- T022 → T023 → T024 → T025 → T026 → T027
- T031 → T032 → T033 → T034 → T035 → T036 → T037
- T041 → T042 → T043 → T044 → T045
- T049, T050 → T051 → T052 → T053 → T054 → T055

---

## Parallel Examples

### Foundational

```text
Task: "T010 contract validation tests in tests/unit/test_supervisor_searcher_contracts.py"
Task: "T015 VectorIndex.documents in storage/vector_index.py and chroma_vector_index.py"
Task: "T018 quote helpers in source_verification.py and evidence_index.py"
```

### User Story 1

```text
Task: "T019 tests/unit/test_result_evaluation.py"
Task: "T020 tests/unit/test_researcher_stop.py"
Task: "T021 tests/integration/test_worker_mcp.py"
Task: "T022 result-evaluation prompt and output model"
```

### User Story 3 and User Story 4 in parallel (after US2 and Foundational)

```text
Developer A: T038–T046 (gap analysis and rounds)
Developer B: T047–T055 (claim gate)
```

---

## Implementation Strategy

### MVP (User Story 1)

1. Phases 1–2.
2. Phase 3 (US1): searchers keep only good evidence and stop for a reason; reports still use the
   interim summary, so the rest of the workflow runs unchanged.
3. Validate with quickstart scenarios 1–3, then the full suite.

### Incremental delivery

1. US1 → US2 (verified short reports) → US3 (rounds and follow-ups): each keeps the graph runnable.
2. US4 can ship any time after Foundational.
3. US5 last, then Polish.

## Notes

- Nodes return only the keys they change (state has no reducers).
- Every model output passes code validation before it changes state; invalid parts degrade to
  `missing` or a refusal, never to a crash.
- Commit after each task or logical group, one-line subject messages.
