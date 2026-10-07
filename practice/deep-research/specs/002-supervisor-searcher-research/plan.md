# Implementation Plan: Supervisor–Searcher Research Workflow

**Branch**: `002-supervisor-searcher-research` | **Date**: 2026-10-07 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `specs/002-supervisor-searcher-research/spec.md`

## Summary

Evolve the existing research supervisor and research sub-agent into a supervisor–searcher
workflow, without replacing the graph, storage or model layers:

- **Planner** writes 1–4 evidence points per task, each linked to the brief's required answers;
  they are part of the approved plan's hash.
- **Searcher** (today's `ResearchAgent`) evaluates every search result (accept or reject with a
  reason), steers its next query to uncovered evidence points, stops on coverage, budget, or no
  progress, persists accepted pages (every chunk indexed), closes its own gaps with a hybrid
  passage lookup over the run's stored pages, and returns a short `SearcherReport` whose answered
  points carry verbatim quotes that code verifies.
- **Supervisor** keeps its model-driven delegation loop per round. A new graph node,
  `analyze_gaps`, runs after every round: it classifies gaps deterministically (essential = linked
  to a required answer), looks up still-missing essential points across the whole run, and lets
  the supervisor model write bounded follow-up tasks that start the next round.
- **Verification**: a claim-level judge (one sentence = one claim, judged only against the
  passages recorded for the draft, every cited source must support part of the sentence) becomes
  the faithfulness gate. RAGAS Faithfulness moves to the informational quality step.

**Terminology**: *searcher* (spec) = `ResearchAgent` with model role `researcher` (code);
*supervisor* = `ResearchSupervisor`.

## Technical Context

**Language/Version**: Python ≥ 3.11 (local environment runs 3.14)

**Primary Dependencies**: LangGraph (workflow, checkpoints), LangChain core and providers (Google
GenAI, OpenAI, Groq) through `ModelRouter`, Chroma (vector index), MCP client (Tavily), RAGAS
(informational metrics), Pydantic v2 (contracts), Streamlit (UI), Langfuse (tracing). No new
third-party dependency: keyword ranking (BM25) and rank fusion are implemented in-house
(research.md R6).

**Storage**: run-scoped VFS (`storage/vfs.py`) for every artifact; Chroma persistent collection
per embedding model for passages; LangGraph SQLite checkpointer for state; SQLite run registry.

**Testing**: pytest (`tests/unit`, `tests/contract`, `tests/integration`, `tests/ui`,
`tests/evaluation`), offline by default with `ControlledMCP`, `LocalVectorIndex` and fake routers;
ruff (`E`, `F`, `W`).

**Target Platform**: local desktop (Windows, macOS, Linux) running Streamlit.

**Project Type**: single Python application (library package + Streamlit UI).

**Performance Goals**: report writing for ≤ 5 tasks no slower than 20% over today's baseline
(spec SC-007); ≥ 20% fewer web searches per task on the benchmark (SC-008); each passage lookup
completes in about a second for a run of a few hundred chunks.

**Constraints**: every loop bounded (search ≤ 5 per task, accepted pages ≤ 5, supervisor turns
≤ 8 per round, gap-analysis follow-up loop ≤ 4 turns, follow-ups ≤ 2 per required answer and
≤ 4 per run, 2 follow-up rounds without progress stop); no credentials in artifacts, logs or traces; checkpoint resume never repeats completed
searches, captures or reports; searcher report ≤ 400 words, quotes ≤ 30 words, ≤ 5 findings.

**Scale/Scope**: up to 5 planned tasks plus up to 4 follow-up tasks per run; up to 5 accepted
pages per task, each up to 60,000 characters (about 60 chunks of 1,000 characters); a run holds
at most a few thousand chunks.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | How this design complies | Status |
| --- | --- | --- |
| I. Validated Human Input | No new human input path. Plan approval now covers evidence points (hash-bound). Every model output that affects state (result evaluations, searcher reports, follow-up tasks, claim verdicts) is validated by Pydantic contracts and code rules (caps, links, quotes, known ids) before it is used. | Pass |
| II. Explicit, Resumable Workflow | Rounds become explicit graph transitions: `researching → analyze_gaps → researching` or `→ research_complete`, with bounded follow-ups and a no-progress stop. Each node is checkpointed; completed tasks are restored from VFS, never repeated. | Pass |
| III. Durable Research Artifacts | Result evaluations, lookups, searcher reports, gap analyses, follow-up tasks, per-draft evidence and claim evaluations are VFS artifacts with stable paths and ids (contracts/vfs-artifacts.md). | Pass |
| IV. Evidence-Linked Research | Search goes only through `MCPResearchPort`. Only accepted sources become evidence. Every answered point carries a verified verbatim quote. Every report sentence must be supported by its own cited, recorded passages. | Pass |
| V. Verified, Human-Approved Publication | Claim-level gate is stricter than today's score; any new draft still loses authority (`invalidate_draft_authority`). Publication approval unchanged. | Pass |
| Constraints: secrets | All new error paths use `redact_error`; lookups and evaluations store no request URLs. | Pass |
| Delivery gates (tests) | Plan lists tests for rejected input (invalid reports, bad follow-ups), legal/illegal transitions (gap routing), retry exhaustion (caps, stalls), resume, persistence, evidence links (quotes), verification failure (claims). | Pass |

No violations; Complexity Tracking stays empty.

**Post-design re-check (after Phase 1)**: still passes. The design adds one graph node and keeps
every new loop bounded by caps in state ([data-model.md](data-model.md)); every new artifact is
listed in [contracts/vfs-artifacts.md](contracts/vfs-artifacts.md); every model output is
validated by the rules in [contracts/](contracts/) before it changes state; no new external
integration or dependency is introduced.

## Project Structure

### Documentation (this feature)

```text
specs/002-supervisor-searcher-research/
├── plan.md              # This file
├── research.md          # Phase 0: decisions and alternatives
├── data-model.md        # Phase 1: entities, fields, rules, state
├── quickstart.md        # Phase 1: validation scenarios
├── contracts/           # Phase 1: internal interface contracts
│   ├── searcher-report.md
│   ├── supervisor-tools.md
│   ├── passage-lookup.md
│   ├── claim-evaluation.md
│   └── vfs-artifacts.md
└── tasks.md             # Phase 2 (/speckit-tasks, not created here)
```

### Source Code (repository root)

```text
src/intelligence_collector/
├── contracts.py                 # + EvidencePoint, ResultEvaluation, SearcherReport,
│                                #   GapAnalysis, ClaimEvaluation, StopReason, TaskOrigin;
│                                #   TodoPlan / ResearchTodo gain evidence points
├── agents/
│   ├── planner.py               # evidence points in write_todos; link validation; plan_record
│   ├── researcher.py            # search() only: query steering, stop reasons
│   ├── result_evaluation.py     # NEW: per-row accept/reject decisions (code + model)
│   ├── search_task.py           # NEW: SearchTask, the task pipeline in resumable steps
│   ├── searcher_report.py       # NEW: gap self-lookup, report drafting, quote check, limits
│   ├── supervisor.py            # per-round delegation; drives SearchTask steps
│   ├── gap_analysis.py          # NEW: deterministic classification, run-wide lookup,
│   │                            #   follow-up validation (CreateFollowUp)
│   ├── claim_judge.py           # NEW: sentence claims, per-citation support verdicts
│   ├── worker.py                # persist pages first; write evidence bundle after the report
│   ├── prompts.py, tools.py     # result evaluation, report, query rewrite, follow-up, claim prompts
│   └── passage_selector.py      # unchanged (still selects passages while searching)
├── storage/
│   ├── passage_lookup.py        # NEW: multi-query hybrid lookup (vector + BM25, RRF, top-5)
│   ├── vector_index.py          # VectorIndex gains documents(run_id); LocalVectorIndex too
│   └── chroma_vector_index.py   # documents(run_id) via collection.get; chunk_index metadata
├── graph/
│   ├── nodes/research.py        # researching per round; NEW analyze_gaps node
│   ├── nodes/report.py          # verifying uses claim judge; RAGAS moves to evaluating_quality
│   ├── nodes/common.py          # Node.ANALYZE_GAPS; invalidation covers claim fields
│   ├── routing.py, build.py     # route_after_delegation → analyze_gaps; route_after_gap_analysis
│   ├── state.py, migration.py   # research_round, gap_stall_count, follow_up_count, claim fields
├── source_verification.py       # quote normalization helpers; per-draft evidence snapshot load
├── observability/
│   └── supervisor_benchmark.py  # NEW: SC-002/004/005 scoring on fixtures, baseline comparison
└── ui/                          # evidence.py readers + html.py fragments: evaluations, stop
                                 #   reasons, rounds, follow-ups, claim verdicts with cited text

scripts/
└── benchmark_supervisor_searcher.py  # NEW: live, opt-in benchmark runner (baseline + results)

evals/supervisor_searcher/            # NEW: benchmark fixtures, baseline.json, benchmark_results.json

tests/
├── unit/          # contracts, planner links, result evaluation, stop reasons, quote check,
│                  #   report limits/corrections, passage lookup (BM25, RRF), gap analysis,
│                  #   claim judge rules
├── contract/      # tool schemas, structured output schemas, VFS artifact shapes
├── integration/   # round loop with ControlledMCP, follow-up caps and stall stop, resume,
│                  #   verification with claim gate
└── ui/            # activity rendering of new records
```

**Structure Decision**: keep the single-package layout. New behavior lands in three new agent
modules (`searcher_report`, `gap_analysis`, `claim_judge`) and one storage module
(`passage_lookup`); existing modules change only at their seams. The graph gains one node
(`analyze_gaps`) and one router.

## Phases (for /speckit-tasks)

1. **Contracts and plan**: evidence points in `TodoPlan`/`ResearchTodo`, planner validation, plan
   hash; checkpoint migration for legacy todos.
2. **Searcher**: result evaluation and stop reasons; search/report split; persistence ordering;
   passage lookup; searcher report with quote verification and limits; evidence bundle derived
   from the report.
3. **Supervisor rounds**: `analyze_gaps` node, run-wide lookup, `CreateFollowUp` tool, caps and
   stall stop, readiness from gap analysis.
4. **Verification**: per-draft evidence snapshot; claim judge gate; RAGAS to quality step; repair
   wiring; state invalidation of claim fields.
5. **Observability and UI**: activity records, traces, docs and spec sync.

Each phase is independently testable offline; P1 user stories map to phases 2–4.

## Implementation Notes

Where the code differs from the design above (recorded at the end of `/speckit-implement`):

- **Search/report split**: instead of `ResearchAgent.report()`, the task pipeline is
  `SearchTask` (`agents/search_task.py`). The supervisor runs its model steps (phrasings, search,
  report) on worker threads and its storage steps (lookups, persistence, bundle) on its own
  thread, so index reads and writes never overlap. `ResearchAgent.summarize` and `research` are
  removed; the searcher report is saved as soon as it is drafted, so a resume never drafts it
  again.
- **Result evaluation** lives in its own module, `agents/result_evaluation.py`.
- **Gap closure** first checks the reports: an essential gap that a later follow-up's report
  answered is closed (`covered_by_report`) before any lookup (FR-017a).
- **Claim judge**: a sentence that cites a source is always a factual claim (FR-022), so an
  unknown or unsupporting citation can never pass as "framing".
- **Model roles**: `research_compressor` is retired. Old `.env` files that still override it are
  accepted and ignored (`RETIRED_ROLES` in `models/token_limits.py`).
- **UI gate**: the faithfulness gate card shows the claim-support share and passes only at
  100 %; the 0.90 marker is gone. RAGAS faithfulness is shown with the quality signals.
- **Benchmark**: metric code is shared by the offline test and the live runner
  (`observability/supervisor_benchmark.py`). The live runner scores the result evaluator, gap
  analysis and claim judge on the labeled fixtures with live models, and compares the
  end-to-end figures with `baseline.json`. The baseline was recorded on one question
  (`--only sqlite-postgres-mobile`, gpt-5.4-nano), where every task already used a single
  search, so it cannot show the SC-008 reduction.

## Complexity Tracking

No constitution violations to justify.
