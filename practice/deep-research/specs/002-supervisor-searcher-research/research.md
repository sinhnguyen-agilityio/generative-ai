# Research: Supervisor–Searcher Research Workflow

Phase 0 decisions. Each entry: decision, rationale, alternatives considered. Everything here is
grounded in the current code (paths relative to `src/intelligence_collector/`).

## R1. Where the round loop lives

**Decision**: Add a graph node `analyze_gaps` (agent: Research Supervisor) between `researching`
and `research_complete`. `researching` runs the supervisor's existing model-driven delegation loop
for the current round's pending tasks; `analyze_gaps` classifies gaps, looks them up, and either
appends follow-up tasks (route back to `researching`) or ends research (route to
`research_complete`). A replan proposal still routes `researching → replan`.

**Rationale**: Constitution II requires LangGraph to own transitions and checkpoints. A round per
node pass gives a checkpoint after every round, so a crash mid-research resumes at the right round
without replaying completed tasks (`already_researched` already guards tasks). It also keeps the
supervisor's delegation loop unchanged in shape.

**Alternatives**: rounds inside `ResearchSupervisor.delegate` (no checkpoint between rounds, resume
must reconstruct rounds from VFS; rejected); a separate supervisor subgraph (more wiring, no extra
benefit).

## R2. Evidence points in the plan

**Decision**: `TodoPlan` (the `write_todos` tool arguments) gains `evidence_points: list[list[
{text, required_answers: list[int]}]]`, aligned with `questions`; `required_answers` are 1-based
indexes into the brief's checklist. Code validates 1–4 points per task, valid indexes, and that
every checklist item is linked at least once; a violation is returned to the planner as a
`ToolMessage` error and retried within `max_attempts` (existing loop in `PlannerAgent`).
`ResearchTodo.evidence_points` stores `EvidencePoint(point_id="E1"…, text, required_answers=[texts])`.
`plan_record` adds evidence points to the hashed task body. `evidence_needed` stays as a joined
rendering for prompts and backward compatibility.

**Rationale**: indexes are robust to model paraphrasing; storing the linked texts keeps the todo
self-describing after the brief changes.

**Legacy checkpoints**: a todo without evidence points gets one derived point at runtime
(`evidence_needed` or the question) linked to every required answer, so it is treated as
essential (conservative: more follow-ups, never fewer). The stored plan and its hash are not
rewritten.

## R3. Result evaluation in the searcher

**Decision**: After each search, deterministic checks run first: duplicate URL (already seen by
this task) → `duplicate`; no usable page and empty snippet → `no_usable_content`. The remaining
rows (with their selected passages) go to one structured call (role `result_evaluator`, standard
tier) returning, per row, `accept` or a rejection reason (`off_topic`, `low_credibility`,
`outside_scope`) and the evidence point ids the row covers. Credibility guidance in the prompt:
prefer primary/official sources, recency for time-sensitive points, and consistency with other
sources; no allow/block lists. URL claiming (`UrlClaims`) happens only for accepted rows; a new
`UrlClaims.release(url, task_id)` is not needed because claims move after the evaluation. A row
accepted without a readable page is stored as snippet-only (`snippet_only = true`). If the
evaluation call fails, no row of that search is accepted and the search counts as a failed search
(toward the `search_failures` stop).

**Rationale**: the cheap deterministic rules remove obvious cases without a model call; one batched
call per search keeps cost bounded (≤ 5 calls per task). Covered point ids drive query steering
and the "all points covered" stop.

**Alternatives**: let the researcher model decide inside its tool-calling turn (unstructured, not
testable); evaluate each row separately (5× the calls).

## R4. Query steering and stop reasons

**Decision**: The researcher prompt receives, after every search, the list of uncovered evidence
points and the queries already used. The tool node refuses a query whose normalized form (casefold,
collapsed whitespace, sorted words) repeats an earlier query of the task. `route_after_llm`
stops with a recorded `StopReason`: `all_points_covered`, `search_budget_spent`,
`page_budget_full`, `no_progress` (2 consecutive searches with no accepted row),
`search_failures` (2 consecutive failed searches, including failed result evaluations), or
`model_finished` (the model ended without a tool call). The first search is still forced
(existing behavior). Report problems are not stop reasons: they set `report_status` (R7).

**Rationale**: deterministic stops are testable; steering only changes prompt content.

## R5. Search → persist → report split and persistence ordering

**Decision**: Split `ResearchAgent.research` into `search()` (the existing LangGraph loop without
compression) and `report()` (gap self-lookup + report). In
`ResearchSupervisor._gather_sub_agents`, each job runs `search` in a worker thread, then persists
accepted pages on the event-loop thread (serialized, as today), then runs `report` in a worker
thread, all under the same concurrency slot. The worker writes raw captures, indexes **every chunk
of the cleaned page** (already the case: `source_chunks(text=page)`), then, after the report, the
evidence bundle (`evidence/result.json`). The bundle is written last, so `already_researched`
still means "task fully done".

**Rationale**: meets FR-009 (persist before reporting) and FR-013b (lookups see the task's own
pages), keeps Chroma and `UrlRegistry` writes single-threaded, and keeps parallelism for the slow
parts.

**Alternatives**: persist inside the searcher thread (Chroma/registry concurrency risk); report on
the event-loop thread (serializes LLM calls).

## R6. Passage lookup (hybrid, top-5)

**Decision**: New `storage/passage_lookup.py`:

1. Queries: the evidence point text + " — " + task question, plus 2–3 phrasings from one batched
   call per task (role `query_rewriter`, standard tier), instructed to include at least one
   phrasing in the language of the source titles shown. On failure, the combined query alone.
2. Candidates: vector search per query through `VectorIndex.retrieve(query, limit=10,
   run_id=...)`; keyword search with an in-house Okapi BM25 (k1 = 1.5, b = 0.75, lowercase
   `\w+` tokens) over `VectorIndex.documents(run_id)`.
3. Restrict to allowed source ids: the sources of the run's active tasks (the catalog's active
   todo ids). For the searcher's self-lookup that means its own accepted sources plus those of
   active tasks that have finished; pages captured for a superseded plan are never eligible.
4. Fuse with Reciprocal Rank Fusion (k = 60) across all queries and both rankings; return the top
   5 chunks; no neighbor expansion; no score threshold.
5. If BM25 cannot run (no documents API), fall back to vector ranking and record it.

`VectorIndex` gains `documents(run_id) -> list[IndexedChunk]`; Chroma implements it with
`collection.get(where={"run_id": run_id})`, `LocalVectorIndex` with its JSON rows. Chroma upserts
also store `chunk_index` (today dropped) for traceability.

**Rationale**: no new dependency (`rank_bm25`/`langchain_community` are not installed); BM25 over
a few thousand chunks is fast in pure Python; RRF needs no score calibration.

**Alternatives**: Chroma full-text `where_document` (`$contains` is substring filtering, not
ranking); LangChain `EnsembleRetriever` (import path changed in LangChain 1.x, extra dependency);
model re-ranking (rejected in clarification).

## R7. Searcher report drafting and validation

**Decision**: One structured call (role `searcher_report`, standard tier; replaces
`research_compressor`) reads the passages selected while searching plus lookup results for
uncovered points and returns `SearcherReportDraft` (per point: status, sources with quotes; ≤ 5
findings with source ids). Code then:

- verifies every quote against the cited source's stored cleaned page text (VFS `raw/<n>.md`, or
  the snippet for snippet-only sources) after normalization (NFKC, casefold, collapsed whitespace,
  unified quotes and dashes); a point with no verified quote becomes `missing`;
- drops sources that are not accepted sources of the run, findings citing them, and fills absent
  points as `missing`, recording each correction (FR-013c);
- enforces limits: quotes cut to 30 words (only at a prefix that still verifies), findings dropped
  from the end until ≤ 400 words, never dropping a point's status, source or quote;
- fills counts, queries, stop reason and failure kind from the search phase.

`report_status` is `complete`, or `corrected` when any correction was applied. If the call fails
or returns nothing readable, every point is `missing` and `report_status = unreadable`. The evidence bundle is derived from the report: `summary` = findings,
`sufficient` = every point answered, `unresolved_gaps` = missing/partial points, and each
source's `excerpt` = selected passages plus the looked-up passages that support a quote (so the
writer sees what the report relied on).

**Rationale**: the bundle shape stays compatible with readiness, the catalog and the writer.

## R8. Gap analysis and follow-up tasks

**Decision**: `agents/gap_analysis.py`, called by the `analyze_gaps` node:

1. Collect `missing`/`partial` points from all reports of the run's active tasks.
2. Classify deterministically: essential if `required_answers` is non-empty.
3. For each essential gap, run the passage lookup over every source of the active catalog; one
   structured call (role `supervisor`, reasoning tier) per round judges all gaps at once and
   returns, per gap, `covered` with source id + quote, or `uncovered`. Code verifies quotes as in
   R7.
4. If essential gaps remain and caps allow, the supervisor model is asked, in a loop of at most 4
   turns per analysis, to write follow-ups via the `CreateFollowUp` tool (question, 1–4 evidence points linked to required answers, the gap
   point ids it targets, reason). Code refuses: unknown or already-covered gap ids, links outside
   the brief's required answers, more than 2 follow-ups per required answer or 4 per run, and
   duplicates of an earlier follow-up question (normalized). A gap the model says is out of scope
   is answered with `ProposeReplan` (existing tool).
5. Record `GapAnalysis` (round, gaps, classification, lookups, coverage judgments, follow-ups,
   refusals, turns used). Stall rule: only follow-up rounds (round ≥ 2) count; if such a round
   closed no essential gap, `gap_stall_count += 1`; at 2, stop. When the 4 turns are spent the
   analysis ends with the follow-ups accepted so far, sets `turn_limit_reached = true`, and
   `ended = turn_limit` if none was created.
6. Same-task second attempts: `ConductResearch` for an already researched task is refused unless
   its latest report has `failure_kind = technical`; gap analysis reads the latest attempt's
   report.

Follow-up tasks are appended to `todos` with `origin="follow_up"`, `round`, `gap_point_ids`. They
are excluded from the plan hash (recorded under `plan["follow_ups"]`), so the existing approval
check in `researching` keeps passing.

**Rationale**: deterministic where possible (classification, caps, refusals); one reasoning call
per round for semantic judgment.

## R9. Readiness

**Decision**: `research_complete` keeps reading evidence bundles but takes readiness from the
latest `GapAnalysis`: READY when no essential gap remains; PARTIAL when only non-essential or
capped/stalled essential gaps remain and usable evidence exists; INSUFFICIENT when no usable
evidence exists. The existing one-time gap research (`prepare_gap_research`) remains only for
INSUFFICIENT, since essential gaps were already handled by follow-ups.

## R10. Per-draft evidence snapshot

**Decision**: `synthesizing` writes `synthesis/evidence.v{N}.json`: for every source id shown to
the writer, the exact text shown (title + excerpt as truncated by `EXCERPT_CHARS`), with the
question it was shown under. A presentation-only repair (writer sees no excerpts) copies the
previous draft's snapshot. The claim judge and quality metrics read only this snapshot.

**Rationale**: FR-023 "only the passages the writer was shown" must be reproducible after resume.

## R11. Claim-level judge

**Decision**: `agents/claim_judge.py`:

- Segment the scored body with the existing `report_sections` rules (unscored sections, title,
  gap sentences and `Assessment:` sentences excluded — the existing `ANALYSIS_PREFIX` is the
  analysis label), then split into sentences (period/question/exclamation boundaries outside
  citations and numbers; each list item is a sentence). Citations are parsed per sentence.
- Batch sentences (≤ 10 per call, ≤ 4 calls in flight) to role `claim_judge` (reasoning tier, a
  model separate from the writer's tier). Input per sentence: the sentence and, for each cited
  source, its snapshot text. Output per sentence: `states_fact`, verdict (`supported`,
  `partly_supported`, `unsupported`, `contradicted`), per cited source `supports_part`, and the
  unsupported parts.
- Code applies FR-024/FR-026: a factual sentence without citations fails; an unknown source id
  fails; any cited source with `supports_part = false` fails naming it; any verdict other than
  `supported` fails. Share supported = supported / factual sentences.
- A judge failure or a missing verdict after one supplemental call is ERROR (existing retry path).

The gate replaces RAGAS in `verifying`: failing claims become `VerificationCheck(gate=
"faithfulness", missing_points=[sentences], affected_ids=[...])`, which `repair.diagnose` already
routes to `vfs`. RAGAS Faithfulness runs in `evaluating_quality` on the passing draft, uses the
same snapshot as contexts, and never blocks (FR-026a). State: new `claim_support_score` and
`claim_evaluation_ref`; `faithfulness_score` keeps meaning RAGAS; `SCORE_FIELDS` moves the
faithfulness producer to `evaluating_quality` and adds `claim_support`.

**Alternatives**: atomic-fact claims (rejected in clarification); RAGAS-only (no per-citation
check).

## R12. Model roles and limits

New roles in `ModelFactory`/`token_limits`: `result_evaluator` (standard, 1024 tokens),
`query_rewriter` (standard, 512), `searcher_report` (standard, 1536; replaces
`research_compressor`), `claim_judge` (reasoning, 3072). Supervisor gap judgment reuses
`supervisor`. New settings: `INTELLIGENCE_MAX_SOURCES_PER_TASK` default 5 (still ≤ 10),
`INTELLIGENCE_LOOKUP_TOP_K` (5, 1–20), `INTELLIGENCE_MAX_FOLLOW_UPS_PER_RUN` (4, 0–8),
`INTELLIGENCE_MAX_FOLLOW_UPS_PER_ANSWER` (2, 0–4), `INTELLIGENCE_REPORT_MAX_WORDS` (400).

## R13. Benchmark set

**Decision**: two layers. A live, opt-in runner `scripts/benchmark_supervisor_searcher.py`
(`--run-live` plus model and Tavily keys) runs the benchmark questions end to
end; it is run once on the unchanged code to write `evals/supervisor_searcher/baseline.json`
(repair-cycle share, searches per task) and again after the feature to write `benchmark_results.json`.
Offline, `tests/evaluation` checks the metric computations on fixtures with scripted routers.
Fixtures: add `evals/supervisor_searcher/` with offline fixtures (ControlledMCP pages with
labeled relevant/off-topic/duplicate results, planted essential gaps, and drafts with planted
unsupported and mis-cited sentences) used by `tests/evaluation`; live runs reuse the same
questions. Required for SC-002, SC-004, SC-005, SC-007, SC-008.
