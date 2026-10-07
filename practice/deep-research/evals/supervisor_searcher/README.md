# Supervisor–searcher benchmark (spec 002)

Fixtures and results for the success criteria of
[specs/002-supervisor-searcher-research](../../specs/002-supervisor-searcher-research/spec.md).

| File | Used by | Purpose |
| --- | --- | --- |
| `questions.json` | `scripts/benchmark_supervisor_searcher.py` | Fully scoped questions run live end to end |
| `results.json` | offline tests | Search rows labeled relevant / off_topic / duplicate / no_usable_content / low_credibility (SC-002) |
| `gaps.json` | offline tests | Plans and round-1 reports with planted essential and non-essential gaps (SC-004) |
| `claims.json` | offline tests | Report sentences with planted claim errors and their evidence snapshot (SC-005) |
| `baseline.json` | SC-007, SC-008 | Live run of the **unchanged** workflow, recorded before the feature |
| `benchmark_results.json` | SC-002…SC-008 | Live run after the feature |

## Running

```text
.venv\Scripts\python.exe scripts/benchmark_supervisor_searcher.py --run-live --baseline
.venv\Scripts\python.exe scripts/benchmark_supervisor_searcher.py --run-live
```

`--run-live` authorizes paid model, embedding and Tavily calls. The runner uses scratch stores
under `.tmp/b/<HHMMSS>/` (short, for Windows path limits), answers plan approval with "approve", stops at the
publication-approval (or recovery) pause and never publishes. A clarification pause is recorded
as `skipped`. Errors are stored redacted.

If the model or Tavily settings are missing, the runner writes `{"live": false, "missing": [...]}`;
SC-007 and SC-008 then cannot be compared against a baseline.
