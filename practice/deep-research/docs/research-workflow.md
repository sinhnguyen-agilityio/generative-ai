# Research workflow

This describes the graph in `src/intelligence_collector/graph/build.py` as implemented on the
`feat/deep-research` branch. An earlier, larger refactor of this workflow (durable operations,
technical-recovery node, cycle ledger) is not part of this code; it exists only in
`git stash@{0}` ("refactor code").

## Shortest successful path

```text
clarify_with_user -> write_research_brief -> planning -> await_plan_approval (interrupt)
  -> researching -> research_complete -> synthesizing -> verifying --PASS--> evaluating_quality
  -> approval (interrupt) -> publishing
```

1. The brief keeps the user's question: `original_question` (system-owned, the user's own turns),
   `research_question`, `required_answers` (the acceptance checklist) and
   `output_requirements` (only explicitly stated language/length/format).
2. The planner writes the fewest tasks that work (one for a simple question or comparison),
   each with `evidence_needed`, plus `report_shape`. `write_todos` persists the todos and
   `planning/plan.v<n>.json` persists the hash-bound plan.
3. `await_plan_approval` pauses before any retrieval. Approve → research; "cancel" → the run
   ends; any other reply → rescoping and a new plan that needs its own approval.
   `researching` refuses to run unless the approval matches the current plan hash.
4. The research supervisor (an LLM) delegates the tasks with `think_tool`, `ConductResearch`,
   `ResearchComplete` and `ProposeReplan`: one research sub-agent per call, at most three at once
   (`INTELLIGENCE_MAX_CONCURRENT_RESEARCHERS`), plus at most one follow-up per task for a named
   gap. Each sub-agent searches with `tavily_search`, reflects with `think_tool`, and reads the
   three passages of every fetched page closest to its question (chosen by embedding similarity,
   kept verbatim) instead of the page itself. Full
   pages (`raw/<n>.md`), raw captures and provenance are saved in VFS and the pages are indexed
   as chunks; nothing already researched is searched again.
5. The writer receives the original question, required answers, output requirements, report
   shape, the source catalog with stable `[S<n>]` ids, and known gaps. It starts with `## Answer`
   (a direct conclusion for each required item), cites every claim, words a gap as "could not be
   established from the collected sources", and labels its own judgment `Assessment:`.
   A question's stored excerpts up to 12 000 characters are read directly; only a larger
   corpus is narrowed with Chroma.
6. `verifying` runs exactly three gates on the exact draft and stops at PASS:
   1. **Coverage** (LLM judge, criteria with ids `[C1]…`): each required answer must be
      *answered*, not merely mentioned; an "could not be established" statement passes only when
      the evidence really lacks the answer. Output requirements, report guidelines and reviewer
      requests are checked here too. Failed items carry the item text and `missing_points`.
   2. **Citation resolution**: every `[S<n>]` exists among sources collected in this run and
      has a matching URL in `## Sources`.
   3. **RAGAS Faithfulness** strictly greater than 0.90, scored on cited source titles and
      excerpts. Title lines, gap statements and `Assessment:` sentences are not scored (the
      coverage gate judges them).
7. A passing first draft goes straight to report approval: no improvement cycle runs.

## Verification outcomes

| Outcome | Meaning | What happens |
|---|---|---|
| PASS | All three gates passed | Optional RAGAS observations, then report approval |
| FAIL | A gate found a report defect | `fixing` opens an improvement cycle (max `INTELLIGENCE_RECOVERY_ATTEMPTS`, default 3) |
| ERROR | No trustworthy verdict (judge omitted criteria after one supplemental call, evaluator exception, missing/NaN score, evaluator not configured) | One retry of the same draft reusing saved coverage checks, then `needs_review`; the report is never rewritten |

## Repair diagnosis (`repair.py`, no model call)

| Failed check | Route | Correction |
|---|---|---|
| Answer in stored evidence but not in the report; unknown citation id; unsupported statement | `vfs` | Rewrite only the affected answers from the full stored evidence |
| Evidence gap, but the task has stored sources the writer did not see | `chroma` | Locate passages with Chroma, read their VFS captures through provenance |
| Evidence gap, nothing unseen | `research` | Reopen only the affected task with the missing points (targeted Tavily MCP search) |
| Evidence gap after targeted research already ran for that item | `replan` | Replacement tasks → `await_plan_approval` again |
| Language, length, format, citation layout | `presentation` | Edit presentation only |

The cycle takes the largest route needed; the writer receives every issue plus the answers that
already passed (to keep them) and the previous draft. Every new draft clears verification,
report approval and publication. Repeating the same failure twice ends in `needs_review`.

## Current node → component mapping (changes in this revision)

| Node | Before | Now | Failure it fixes |
|---|---|---|---|
| `write_research_brief` | One "maximally specific" question | Question + checklist + explicit output requirements + verbatim original | Brief inflated "latest Python version" into "sources must explicitly confirm…", which made a correct answer look unsupported |
| `planning` | Questions only | Plan record with evidence needed and report shape, hash-bound | Plan had nothing to approve; 2–3 overlapping tasks for simple comparisons |
| `await_plan_approval` (new) | — | Interrupt before retrieval | Retrieval started without approval |
| `research_complete` | Assessor LLM call; PARTIAL marked the run incomplete | Readiness from stored bundles; gaps become writer limitations | PARTIAL produced "INCOMPLETE:" answers and skipped report approval |
| `synthesizing` | Reflection lessons + full rewrite | Answer-first writer; targeted repair block | Reports narrated "the provided evidence" and left questions unanswered |
| `verifying` | Fact-check judge, criteria matched by verbatim text | Three gates, id-matched coverage, PASS/FAIL/ERROR | Judge mismatches produced ERROR on correct reports; a report that never stated what 403 means passed |
| `fixing` | Reflection LLM call | Deterministic diagnosis (`repair.py`) | Repairs did not target the failing item; extra model call per cycle |
| `prepare_repair_research`, `replan` (new) | — | Targeted research / replacement plan | Missing evidence could only be "fixed" by rewriting |
| `evaluating_quality` | Faithfulness + optional metrics | Optional metrics only | Faithfulness was observational |
| `finish_incomplete` | Ended incomplete runs without approval | Removed | Honest reports never reached a human |

Graph nodes stay small; durable state lives in checkpoints (plan, approval, repair plan,
history) and VFS (`planning/plan.v<n>.json`, `verification/repair.<n>.json`,
`verification/result.<id>.json`).

## Limitations

- Tavily search returns short snippets (~130–1 500 characters) and, with `include_raw_content`
  (verified on 2026-10-05: ~6 600–7 900 characters per page), each page's cleaned text. Pages are
  capped at 60 000 characters and reach LLMs only as summaries, so a detail the summary drops is
  found through the vector index rather than the writer's prompt.
- Faithfulness requires the judge key (`OPENAI_API_KEY` by default); without it the app shows a
  warning and every verification is ERROR. Offline tests inject fake evaluators.
- Coverage and Faithfulness are LLM judgments and can disagree with a human (see
  `evals/results.md`).
