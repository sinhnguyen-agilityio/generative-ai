# Research workflow

This describes the graph in `src/intelligence_collector/graph/build.py` as implemented on the
`feat/deep-research` branch. An earlier, larger refactor of this workflow (durable operations,
technical-recovery node, cycle ledger) is not part of this code; it exists only in
`git stash@{0}` ("refactor code").

## Shortest successful path

```text
clarify_with_user -> write_research_brief -> planning -> await_plan_approval (interrupt)
  -> researching -> analyze_gaps [-> researching (follow-up round) ...] -> research_complete
  -> synthesizing -> verifying --PASS--> evaluating_quality -> approval (interrupt) -> publishing
```

1. The brief keeps the user's question: `original_question` (system-owned, the user's own turns),
   `research_question`, `required_answers` (the acceptance checklist) and
   `output_requirements` (only explicitly stated language/length/format).
2. The planner writes the fewest tasks that work (one for a simple question or comparison),
   each with `evidence_needed` and 1–4 **evidence points** (`E1`…`E4`, each linked to the
   required answers it serves), plus `report_shape`, by calling the `write_todos` tool. An
   invalid call, including a required answer no point serves, gets its validation error back
   and is retried. `write_todos` persists the todos and `planning/plan.v<n>.json` persists the
   hash-bound plan. A point linked to a required answer is *essential*; the others are not.
3. `await_plan_approval` pauses before any retrieval. Approve button → research; Cancel button →
   the run ends; any typed reply → rescoping and a new plan that needs its own approval.
   `researching` refuses to run unless the approval matches the current plan hash.
4. The research supervisor (an LLM) delegates the tasks with `think_tool`, `ConductResearch`,
   `ResearchComplete` and `ProposeReplan`: one searcher per task, at most three at once
   (`INTELLIGENCE_MAX_CONCURRENT_RESEARCHERS`). A task runs again only after a technical
   failure. Each searcher follows the pipeline in [Searcher pipeline](#searcher-pipeline) and
   returns a short structured report; `analyze_gaps` then looks at the whole round and may
   create follow-up tasks for the next one (see [Research rounds](#research-rounds-and-follow-ups)).
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
   3. **Faithfulness (claim judge)**: every factual sentence of the report body is judged
      against the text of the sources it cites, exactly as the writer saw it (the draft's
      evidence snapshot `synthesis/evidence.v<N>.json`). It fails when a factual sentence is
      uncited, cites an unknown source, cites a source that supports no part of it, is only
      partly supported, unsupported or contradicted. A sentence that cites a source always counts
      as a factual claim. The draft passes only at 100 % support; the share is
      `claim_support_score` and the verdicts are in `verification/claims.<id>.json`. Title lines,
      the Scope, Limitations and Sources sections, gap statements and `Assessment:` sentences
      are not judged (the coverage gate judges them).
7. A passing first draft goes straight to report approval: no improvement cycle runs.
   `evaluating_quality` then scores RAGAS faithfulness, context precision and response
   relevancy as information only: a low score never blocks the report.

## Searcher pipeline

`SearchTask` (`agents/search_task.py`) runs each task in steps. Model steps run on worker
threads; index reads and writes run on the supervisor's thread only.

1. **Phrasings** (`query_rewriter`): 2–3 alternative phrasings per evidence point.
2. **Lookup before search**: each point is looked up in pages that finished tasks already
   stored. When every point has passages, the first web search is no longer forced.
3. **Search**: the searcher calls `tavily_search`; a repeated query is refused. Every result
   row is **evaluated** (`result_evaluator`): duplicates and empty rows are rejected by code,
   the rest are accepted (with the points they cover) or rejected as `off_topic`,
   `no_usable_content`, `duplicate`, `low_credibility` or `outside_scope`. A row accepted on
   its snippet alone is marked `snippet_only`. The search stops with a **stop reason**:
   `all_points_covered`, `search_budget_spent`, `page_budget_full` (`INTELLIGENCE_MAX_SOURCES_PER_TASK`,
   default 5), `no_progress` (two searches in a row with nothing accepted), `search_failures`
   (two failures in a row, evaluator failures included) or `model_finished`.
4. **Persist**: `evaluations.json`, then each accepted page (raw capture, cleaned text) is saved
   and indexed as chunks. A URL already stored in the run is referenced, not stored again.
5. **Lookup before report**: points the search did not cover are looked up again, now
   including this task's own pages.
6. **Report** (`searcher_report`): per evidence point, `answered`, `partial` or `missing`, with
   quotes. Every quote is checked against the stored text of its source (NFKC, case-folded,
   whitespace and quote/dash variants unified); a quote that fails is dropped and the point
   downgraded, and the report status becomes `corrected`. An unreadable draft reports every
   point missing (`unreadable`). The report is saved to `report.json` as soon as it is written.
7. **Finish**: `lookups.json` and the evidence bundle; the bundle is written last and marks the
   task done.

**Passage lookup** (`storage/passage_lookup.py`) ranks chunks of the run's stored pages for the
combined query and each phrasing with the vector index and with an in-process Okapi BM25
(k1 1.5, b 0.75), fuses the rankings with reciprocal rank fusion (k = 60) and keeps the top 5
(`INTELLIGENCE_LOOKUP_TOP_K`). If keyword ranking is unavailable it falls back to
`vector_only`; without phrasings, to `combined_query_only`. Only sources of the run's active
tasks are eligible.

## Research rounds and follow-ups

`analyze_gaps` runs once per round, after every task of the round has finished:

1. Each task's latest report gives its gaps: points not `answered`. A gap on an essential
   point is essential; a non-essential gap becomes a stated limitation.
2. An essential gap is closed when a later follow-up's report answered it, or when a lookup
   across the run's stored pages finds a passage and the supervisor confirms it with a quote
   that is checked against the stored text.
3. For the gaps still open, the supervisor may call `CreateFollowUp` (at most 4 turns): at most
   2 follow-ups per required answer (`INTELLIGENCE_MAX_FOLLOW_UPS_PER_ANSWER`) and 4 per run
   (`INTELLIGENCE_MAX_FOLLOW_UPS_PER_RUN`). A refused call is recorded with its reason.
4. Follow-ups run in the next round. Research ends when no essential gap is open, the caps are
   reached, the supervisor stops, or two rounds after the first made no progress.

Each analysis is saved in `research/rounds/round-<n>.json` (reused on resume, so a resumed run
never asks the model again) with the lookups in `round-<n>.lookups.json`. Follow-ups are listed
in the plan under `follow_ups` but are not part of its approved hash.

## Verification outcomes

| Outcome | Meaning | What happens |
|---|---|---|
| PASS | All three gates passed | Optional RAGAS observations, then report approval |
| FAIL | A gate found a report defect | `fixing` opens an improvement cycle (max `INTELLIGENCE_RECOVERY_ATTEMPTS`, default 3) |
| ERROR | No trustworthy verdict (a judge omitted criteria or sentences after one supplemental call, judge exception, claim judge not configured) | One retry of the same draft reusing saved coverage checks, then `needs_review`; the report is never rewritten |

## Repair diagnosis (`repair.py`, no model call)

| Failed check | Route | Correction |
|---|---|---|
| Answer in stored evidence but not in the report; unknown citation id; a sentence the claim judge failed (the rewrite receives each failing sentence and why) | `vfs` | Rewrite only the affected answers from the un-narrowed catalog (every stored bundle with its selected passages, not the raw pages) |
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
| `researching` (spec 002) | Sub-agent summarized its pages | Searcher pipeline: evaluated results, stop reasons, quote-checked report | Off-topic pages reached the writer; summaries could not be checked |
| `analyze_gaps` (new, spec 002) | Supervisor follow-ups per task | One analysis per round, lookup before follow-up, bounded follow-ups | Follow-ups repeated searches for evidence the run already had |
| `verifying` (spec 002) | RAGAS Faithfulness > 0.90 as the third gate | Claim judge per sentence, 100 % support | A report could pass with a mis-cited sentence |
| `evaluating_quality` | Optional metrics | RAGAS faithfulness, context precision, response relevancy, informational | RAGAS scores were not explainable per sentence |
| `finish_incomplete` | Ended incomplete runs without approval | Removed | Honest reports never reached a human |

Graph nodes stay small; durable state lives in checkpoints (plan, approval, repair plan,
history) and VFS (`planning/plan.v<n>.json`, `verification/repair.<n>.json`,
`verification/result.<id>.json`).

## Limitations

- Tavily search returns short snippets (~130–1 500 characters) and, with `include_raw_content`
  (verified on 2026-10-05: ~6 600–7 900 characters per page), each page's cleaned text. Pages are
  capped at 60 000 characters and reach LLMs only as selected passages, so a detail outside them
  is found through passage lookup rather than the writer's prompt.
- The claim judge runs on the reasoning-tier model; the RAGAS signals need the judge key
  (`OPENAI_API_KEY` by default). Without it the app shows a warning and the report simply has
  no RAGAS signals. Offline tests inject fake judges and evaluators.
- Coverage and claim verdicts are LLM judgments and can disagree with a human (see
  `evals/results.md` and the labeled fixtures in `evals/supervisor_searcher/`).
