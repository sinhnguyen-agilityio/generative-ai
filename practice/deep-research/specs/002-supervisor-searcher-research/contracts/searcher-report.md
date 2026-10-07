# Contract: Searcher Report

Producer: `agents/searcher_report.py` (after the task's pages are persisted). Consumers: the
supervisor (`ConductResearch` tool reply), `analyze_gaps`, `research_complete` (via the derived
evidence bundle), the UI activity view.

## Model output (`SearcherReportDraft`, role `searcher_report`)

```json
{
  "points": [
    {
      "point_id": "E1",
      "status": "answered",
      "support": [{"source_id": "src_3f2a9c1b7d4e", "quote": "only one writer at a time"}]
    },
    {"point_id": "E2", "status": "missing", "support": []}
  ],
  "findings": [
    {"text": "SQLite serializes writes through a single database lock.",
     "source_ids": ["src_3f2a9c1b7d4e"]}
  ]
}
```

## Stored report (`SearcherReport`, after code validation)

```json
{
  "todo_id": "t-1",
  "points": [
    {"point_id": "E1", "status": "answered", "lookup_used": true,
     "support": [{"source_id": "src_3f2a9c1b7d4e", "quote": "only one writer at a time"}]},
    {"point_id": "E2", "status": "missing", "lookup_used": true, "support": []}
  ],
  "findings": [{"text": "…", "source_ids": ["src_3f2a9c1b7d4e"]}],
  "accepted_count": 3,
  "rejected_count": 4,
  "queries": ["SQLite concurrent writes", "SQLite WAL writer lock"],
  "stop_reason": "no_progress",
  "failure_kind": null,
  "report_status": "corrected",
  "corrections": ["E2: quote not found in src_9a…; downgraded to missing"]
}
```

## Rules enforced by code (no model call)

| Rule | On violation |
| --- | --- |
| Exactly one entry per evidence point of the task | Absent point added as `missing`; extra ids dropped; correction recorded |
| `answered`/`partial` need ≥ 1 support entry | Downgraded to `missing` |
| `source_id` is an accepted source of the run | Support entry dropped; finding citing it dropped |
| Quote is verbatim in the source's stored text (normalized: NFKC, casefold, collapsed whitespace, unified quotes/dashes) | Support entry dropped; point downgraded if none left |
| Quote ≤ 30 words | Cut to the longest verifying prefix ≤ 30 words |
| Findings ≤ 5, total ≤ 400 words | Findings dropped from the end; points never dropped |
| Unreadable model output | Every point `missing`, `report_status = unreadable` |
| Any correction above applied | `report_status = corrected` (otherwise `complete`) |

## Supervisor rendering

The `ConductResearch` tool reply is a compact text rendering: header (`Task <id> round <n>:
<status>`), one line per point (`E1 answered [S?] "quote"`), findings, counts and stop reason.
Raw pages never reach the supervisor.
