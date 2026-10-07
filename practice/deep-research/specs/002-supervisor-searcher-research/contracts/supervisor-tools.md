# Contract: Supervisor and Searcher Tools

Tool schemas live in `agents/tools.py` (OpenAI function format). Existing tools keep their names
and arguments unless listed here.

## Searcher (`researcher` role)

| Tool | Change |
| --- | --- |
| `tavily_search(query)` | Unchanged signature. The tool reply now lists, per row, `accepted` or `rejected (<reason>)`, the evidence points still uncovered, and refuses a repeated query with `Refused: repeats an earlier query of this task.` |
| `think_tool(reflection)` | Unchanged. |

## Result evaluation (`result_evaluator` role, structured output)

Input: task question, evidence points (`E1…`), rows (index, title, URL, selected passages).

```json
{"rows": [
  {"index": 1, "decision": "accepted", "covers": ["E1"]},
  {"index": 2, "decision": "rejected", "reason": "off_topic"}
]}
```

Rules: every row index answered once (missing → `rejected` with reason `off_topic`, recorded as a
correction); `reason` only from `off_topic`, `low_credibility`, `outside_scope`; `covers` only known
point ids.

## Query rewriting (`query_rewriter` role, structured output)

```json
{"points": [{"point_id": "E2", "phrasings": ["SQLite concurrent writers", "SQLite write lock"]}]}
```

2–3 phrasings per point, at least one in the language of the listed source titles. Failure →
combined query only.

## Supervisor delegation (`supervisor` role) — existing

A `ConductResearch` call for a task that already has a report is refused unless that report's
`failure_kind` is `technical` (`Refused: content gaps are closed by follow-up tasks.`).

`think_tool`, `ConductResearch(todo_id, research_topic)`, `ResearchComplete()`,
`ProposeReplan(reason, todo_ids)`: unchanged. Applies to the current round's pending tasks.

## Gap analysis (`supervisor` role, in `analyze_gaps`)

### Coverage judgment (structured output)

Input: every essential gap with its lookup results (chunk text, source id).

```json
{"gaps": [
  {"point_key": "t-1:E2", "covered": true, "source_id": "src_3f2a9c1b7d4e",
   "quote": "there can only be one writer at a time", "justification": "states the limit"},
  {"point_key": "t-2:E1", "covered": false, "justification": "no passage gives a mobile benchmark"}
]}
```

`covered` counts only if the quote verifies against that source's stored text.

### `CreateFollowUp` (new tool, offered only when essential gaps remain and caps allow)

The follow-up loop gives the supervisor at most 4 turns per analysis; when they are spent the
analysis ends with the follow-ups accepted so far.

```json
{
  "question": "Read performance of PostgreSQL embedded on mobile or ARM devices",
  "evidence_points": [
    {"text": "A read benchmark of PostgreSQL on mobile or ARM hardware", "required_answers": [2]}
  ],
  "gap_point_keys": ["t-2:E1"],
  "reason": "T2 only found server benchmarks"
}
```

| Rule | Refusal text |
| --- | --- |
| 1–4 evidence points; `required_answers` 1-based indexes into the brief checklist, ≥ 1 per point | `Refused: evidence points must link to required answers 1..N.` |
| `gap_point_keys` are current essential, uncovered gaps | `Refused: <key> is unknown or already covered.` |
| ≤ 2 follow-ups per required answer, ≤ 4 per run | `Refused: follow-up limit reached for <answer>.` |
| Question not a normalized duplicate of an earlier follow-up | `Refused: duplicates follow-up <todo_id>.` |

Out-of-scope gaps use `ProposeReplan` (existing human approval path). Each accepted call creates
a `ResearchTodo(origin="follow_up", round=n+1)` appended to state and to `plan["follow_ups"]`
(outside the plan hash).
