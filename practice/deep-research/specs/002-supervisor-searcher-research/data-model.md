# Data Model: Supervisor–Searcher Research Workflow

Pydantic contracts live in `src/intelligence_collector/contracts.py` (strict `Model` base: unknown
fields rejected). Graph state additions live in `graph/state.py`. VFS locations are in
[contracts/vfs-artifacts.md](contracts/vfs-artifacts.md).

## Enums

| Enum | Values |
| --- | --- |
| `PointStatus` | `answered`, `partial`, `missing` |
| `RejectReason` | `off_topic`, `no_usable_content`, `duplicate`, `low_credibility`, `outside_scope` |
| `StopReason` | `all_points_covered`, `search_budget_spent`, `page_budget_full`, `no_progress`, `search_failures`, `model_finished` |
| `ReportStatus` | `complete`, `corrected`, `unreadable` |
| `FailureKind` | `technical`, `content` (absent when the task produced usable evidence) |
| `TaskOrigin` | `plan`, `follow_up` |
| `ClaimVerdict` | `supported`, `partly_supported`, `unsupported`, `contradicted` |

## EvidencePoint

One specific thing a task must find.

| Field | Type | Rules |
| --- | --- | --- |
| `point_id` | str | `E1`…`E4`, unique within its task; global key is `todo_id:point_id` |
| `text` | str | non-blank, ≤ 300 chars |
| `required_answers` | list[str] | texts of the brief's checklist items it serves; empty = non-essential |

Derived: `essential = bool(required_answers)`.

## TodoPlan (planner tool arguments) — changed

| Field | Change |
| --- | --- |
| `evidence_points` | NEW `list[list[{text, required_answers: list[int]}]]`, same length as `questions`; 1–4 per task; indexes 1-based into the brief checklist |

Validation (planner retry on failure): lengths match, 1–4 points per task, indexes in range, every
checklist index linked by at least one point across tasks.

## ResearchTodo — changed

| Field | Type | Rules |
| --- | --- | --- |
| `evidence_points` | list[EvidencePoint] | NEW; 1–4; legacy todos get one derived point (research R2) |
| `origin` | TaskOrigin | NEW, default `plan` |
| `round` | int | NEW, ≥ 1; plan tasks are round 1 |
| `gap_point_ids` | list[str] | NEW; for follow-ups, the `todo_id:point_id` gaps that prompted it |
| `evidence_needed` | str | kept; rendered from evidence points for prompts |

State transitions (unchanged statuses): `planned → completed | failed`; a follow-up starts
`planned` in the round after the analysis that created it. A second research attempt of the same
task (`research_attempt = 2`) is allowed only when the latest report's `failure_kind` is
`technical`; gap analysis always reads the latest attempt's report.

## ResultEvaluation

One search result's decision for one task. Stored per task, append-only.

| Field | Type | Rules |
| --- | --- | --- |
| `url` | str | normalized for identity |
| `title` | str | |
| `query` | str | the search that returned it |
| `search_index` | int | 1-based search call number |
| `decision` | `accepted` \| `rejected` | |
| `reason` | RejectReason \| None | required when rejected, absent when accepted |
| `covers` | list[str] | evidence point ids the row was judged to cover (accepted only) |
| `linked_to_task` | str \| None | when the page was already owned by another task |
| `snippet_only` | bool | accepted with only the search snippet (no readable page); shown as snippet-only |

Rule: only accepted results count toward the page budget, are stored as evidence, or can be cited.

## SearcherReport

The structured result of one task (contract: [contracts/searcher-report.md](contracts/searcher-report.md)).

| Field | Type | Rules |
| --- | --- | --- |
| `todo_id` | str | known task |
| `points` | list[PointReport] | exactly one per evidence point of the task |
| `findings` | list[Finding] | ≤ 5; each cites ≥ 1 accepted source |
| `accepted_count`, `rejected_count` | int | from result evaluations |
| `queries` | list[str] | in order |
| `stop_reason` | StopReason | |
| `report_status` | ReportStatus | `corrected` when `corrections` is non-empty; `unreadable` when no part of the model output could be read |
| `failure_kind` | FailureKind \| None | |
| `corrections` | list[str] | every automatic fix (quote mismatch, unknown source, missing point, trimming) |

`PointReport`: `point_id`, `status`, `support: list[{source_id, quote}]` (required non-empty for
answered/partial; each quote ≤ 30 words and verified), `lookup_used: bool`.
`Finding`: `text`, `source_ids`.
Size: ≤ 400 words over all text fields.

## PassageLookup

One lookup record (contract: [contracts/passage-lookup.md](contracts/passage-lookup.md)).

| Field | Type |
| --- | --- |
| `lookup_id` | str (uuid) |
| `requested_by` | `searcher:<todo_id>` \| `supervisor:round-<n>` |
| `point_key` | `todo_id:point_id` |
| `queries` | list[str] (combined query first) |
| `allowed_source_ids` | list[str] |
| `results` | list[{chunk_id, source_id, rank, vector_rank, keyword_rank}] (≤ 5) |
| `fallback` | `none` \| `vector_only` \| `combined_query_only` |
| `error` | str (redacted) |

## GapAnalysis

One per round.

| Field | Type | Rules |
| --- | --- | --- |
| `round` | int | ≥ 1 |
| `gaps` | list[Gap] | every missing/partial point of the round's active tasks |
| `follow_ups` | list[str] | todo ids created |
| `refusals` | list[{reason, payload}] | refused CreateFollowUp calls |
| `closed_essential` | int | essential gaps closed in this round vs. the previous analysis |
| `stall_count` | int | consecutive follow-up rounds (round ≥ 2) with `closed_essential == 0`; round 1 never counts |
| `ended` | `no_essential_gaps` \| `caps_reached` \| `stalled` \| `turn_limit` \| `follow_ups_created` | `turn_limit` when the 4 supervisor turns were spent without a follow-up |
| `turns_used` | int | supervisor turns in this analysis, ≤ 4 |
| `turn_limit_reached` | bool | true whenever the 4 turns were spent, whether or not follow-ups were created (FR-019a) |

`Gap`: `point_key`, `text`, `status`, `essential`, `required_answers`, `resolution`
(`covered_by_report` \| `covered_by_lookup` \| `follow_up` \| `limitation`), `covering_source_id`,
`quote`, `lookup_id`, `justification`.

## DraftEvidenceSnapshot

`synthesis/evidence.v{N}.json`: `draft_version`, `sources: {source_id: {title, url, text}}`,
`shown_under: {source_id: [question]}`, `copied_from_version` (presentation repair).

## ClaimEvaluation

Contract: [contracts/claim-evaluation.md](contracts/claim-evaluation.md).

| Field | Type |
| --- | --- |
| `draft_version` | int |
| `artifact_hash` | str |
| `claims` | list[Claim] |
| `supported_share` | float (supported ÷ factual sentences; 1.0 when none) |
| `passed` | bool |

`Claim`: `claim_id` (`K<n>`), `section`, `sentence`, `cited_source_ids`, `states_fact`,
`verdict`, `per_source: [{source_id, supports_part}]`, `unsupported_parts`, `failures` (codes:
`uncited`, `unknown_source`, `source_supports_nothing`, `not_fully_supported`, `contradicted`).

## GraphState additions (`graph/state.py`)

| Key | Type | Purpose |
| --- | --- | --- |
| `research_round` | int | current round, 1 after planning |
| `gap_stall_count` | int | consecutive follow-up rounds (round ≥ 2) without closing an essential gap |
| `follow_up_count` | int | follow-ups created in the run |
| `gap_analysis` | dict \| None | latest `GapAnalysis` |
| `claim_support_score` | float \| None | `supported_share` of the latest evaluated draft |
| `claim_evaluation_ref` | str | VFS path of the latest `ClaimEvaluation` |

Existing `faithfulness_score` and `faithfulness_error` keep their names but now describe the
informational RAGAS run in `evaluating_quality`.

`invalidate_draft_authority()` adds `claim_support_score: None` and `claim_evaluation_ref: ""`.
Planning and scope changes reset `research_round=1`, `gap_stall_count=0`, `follow_up_count=0`,
`gap_analysis=None`. `normalize_checkpoint` adds these defaults for old checkpoints.
