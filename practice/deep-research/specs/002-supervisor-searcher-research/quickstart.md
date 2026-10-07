# Quickstart: Validating the Supervisor–Searcher Workflow

Run from the project root (`practice/deep-research`). Offline scenarios need no API keys.

## Prerequisites

```bash
uv sync --extra test
uv run pytest -q              # baseline must be green before starting
```

Live runs additionally need the model settings (`INTELLIGENCE_REASONING_MODEL`,
`INTELLIGENCE_STANDARD_MODEL`, providers) and a Tavily key in `.env` (see `.env.example`).

## Offline validation scenarios

Each scenario is an automated test added with the feature; names are indicative.

| # | Scenario | Command | Expected |
| --- | --- | --- | --- |
| 1 | Planner links every required answer | `uv run pytest tests/unit/test_planner.py -q` | A plan leaving a required answer unlinked is rejected without writing todos; approved plan hash covers evidence points |
| 2 | Searcher rejects off-topic, duplicate and empty results | `uv run pytest tests/unit/test_result_evaluation.py -q` | Only accepted rows are captured and indexed; each rejection has one reason |
| 3 | Stop reasons | `uv run pytest tests/unit/test_researcher_stop.py -q` | `all_points_covered`, `search_budget_spent`, `page_budget_full`, `no_progress` (2 searches with no accepted result), `search_failures` (2 failed searches; a failed result evaluation counts), `model_finished`; a repeated query is refused |
| 4 | Gap self-lookup finds a deep passage | `uv run pytest tests/unit/test_searcher_report.py -q` | Point answered with a verified quote from a chunk the searcher never read; no web search spent |
| 5 | Invalid report is corrected, not rewritten | same file | Unknown source and missing point become `missing`; corrections recorded |
| 6 | Hybrid lookup ranks exact numbers | `uv run pytest tests/unit/test_passage_lookup.py -q` | A "281 TB" passage enters top-5 through BM25; vector-only fallback recorded |
| 7 | Round loop: essential gap → one follow-up → closed | `uv run pytest tests/integration/test_supervisor_rounds.py -q` | Non-essential gap becomes a limitation; follow-up created only after the whole round |
| 8 | Run-wide lookup prevents a follow-up | same file | Gap covered by another task's page; no follow-up |
| 9 | Caps, turn limit and stall stop | same file | ≤ 2 follow-ups per required answer, ≤ 4 per run; ≤ 4 supervisor turns per analysis; 2 follow-up rounds without progress end research (round 1 never counts) |
| 10 | Resume mid-research | `uv run pytest tests/integration/test_supervisor_rounds.py -k resume -q` | Finished tasks restored; no repeated searches or captures; round number preserved |
| 11 | Claim gate | `uv run pytest tests/unit/test_claim_judge.py -q` | Mis-cited, partly supported, uncited and unknown-source sentences fail; supported passes |
| 12 | Gate integration | `uv run pytest tests/integration/test_bvf_loop.py -q` | Claim failures route to `vfs` repair; RAGAS appears only in quality, never blocks |
| 13 | No credentials anywhere | `uv run pytest tests/unit/test_redaction.py -q` | Lookups, evaluations and reports contain no keys |

Full suite and lint:

```bash
uv run pytest -q
uv run ruff check src tests
```

## Live smoke test

```bash
uv run streamlit run app.py
```

1. Ask a comparison question with 3+ required answers; approve the plan and check each task shows
   1–4 evidence points linked to required answers.
2. During research, the activity view shows per task: queries, accepted/rejected counts with
   reasons, stop reason, and report status; after round 1, the gap analysis lists essential and
   non-essential gaps and any follow-up tasks.
3. Inspect `runtime/vfs/<run_id>/research/<todo_id>/report.json` and
   `research/rounds/round-1.json`.
4. On the verified report, the Verifier card shows per-sentence claim results and the claim
   support share; the RAGAS faithfulness score appears only under quality signals.

## Benchmark (success criteria)

Baseline (once, on the unchanged code, before implementation — task T005):

```bash
uv run python scripts/benchmark_supervisor_searcher.py --run-live --baseline
# writes evals/supervisor_searcher/baseline.json
```

After implementation (live, needs model and Tavily keys):

```bash
uv run python scripts/benchmark_supervisor_searcher.py --run-live
# writes evals/supervisor_searcher/benchmark_results.json
```

Expected: result-evaluation precision ≥ 90% on off-topic/duplicate and recall ≥ 85% on relevant
(SC-002), ≥ 80% of planted essential gaps closed and no follow-up without an essential gap
(SC-004), ≥ 90% of planted claim errors flagged with < 5% false flags (SC-005), repair-cycle share
≥ 25% lower than `baseline.json` (SC-007), searches per task ≥ 20% lower than `baseline.json`
(SC-008).

Offline check of the metric computations:

```bash
uv run pytest tests/evaluation -q -k supervisor_searcher
```
