# Contract: VFS Artifacts

All paths are relative to `<vfs_root>/<run_id>/` and written atomically through `ArtifactStore`.
`<prefix>` is `research/<todo_id>` for the first attempt and
`research/<todo_id>/attempts/<n>` afterwards (existing `research_prefix`).

| Path | Kind | Writer | When | New? |
| --- | --- | --- | --- | --- |
| `planning/plan.v{N}.json` | `research_plan` | planner nodes | each plan | changed: tasks carry `evidence_points`; `follow_ups` list outside the hash |
| `<prefix>/evaluations.json` | `result_evaluations` | searcher | after each search (rewritten whole) | new |
| `<prefix>/raw/<n>.md`, `<prefix>/raw/<n>.json` | `raw_page`, `raw_capture` | worker | persist step, accepted pages only | changed: rejected pages are never captured |
| `<prefix>/retrieved_outcome.json` | `retrieved_outcome` | supervisor | after search, before persist (resume copy) | changed: includes evaluations, queries, stop reason |
| `<prefix>/lookups.json` | `passage_lookups` | searcher | after self-lookup | new |
| `<prefix>/report.json` | `searcher_report` | searcher | after report validation | new |
| `<prefix>/evidence/result.json` | `evidence_bundle` | worker | last step of the task | changed: derived from the report; `already_researched` marker |
| `research/rounds/round-{n}.json` | `gap_analysis` | `analyze_gaps` | end of each round | new |
| `research/rounds/round-{n}.lookups.json` | `passage_lookups` | `analyze_gaps` | end of each round | new |
| `research/supervisor_log.json` | `supervisor_log` | supervisor | each decision | changed: adds `gap_analysis`, `follow_up`, `follow_up_refused` entries |
| `synthesis/evidence.v{N}.json` | `draft_evidence` | `synthesizing` | each draft | new |
| `verification/claims.{verification_id}.json` | `claim_evaluation` | `verifying` | each claim gate run | new |
| `verification/quality.{verification_id}.json` | `quality` | `evaluating_quality` | passing draft | changed: includes RAGAS faithfulness score/error |

Rules:

- No artifact stores credentials, request URLs with keys, or raw provider errors (`redact_error`).
- Rewriting the same path with identical content is a no-op for resume purposes; a task is done
  only when `<prefix>/evidence/result.json` exists.
- Every report, lookup, evaluation and claim record references sources by the run-stable
  `source_id` (`src_…`) or the catalog id (`S<n>`) as documented per contract.
