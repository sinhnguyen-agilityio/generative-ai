# Feature Specification: Supervisor–Searcher Research Workflow

**Feature Branch**: `002-supervisor-searcher-research`

**Created**: 2026-10-06

**Status**: Draft

**Input**: User description: "Implement a supervisor–searcher multi-agent research workflow. The
supervisor delegates research tasks to searcher sub-agents. Each searcher independently searches,
evaluates results, improves its queries, and stops within bounded limits. It persists accepted
source content and evidence in the vector database, then returns a short structured report. The
supervisor analyzes these reports for essential coverage gaps and creates targeted follow-up tasks
when necessary. The final research report is evaluated for claim-level faithfulness, citation
support, and acceptance criteria. Integrate with existing abstractions; no unnecessary rewrite."

## Overview

The run already has a research supervisor that delegates approved tasks to research sub-agents in
parallel, bounded search budgets, a single capture of each page per run, and three verification
gates on the report (coverage of required answers, citation resolution, and an overall
faithfulness score). This feature closes the gaps between that baseline and the requested
behavior:

| Requested behavior | Baseline today | Change |
| --- | --- | --- |
| Searcher evaluates results | Every new result is kept until the page budget is full | Each result is accepted or rejected with a reason; only accepted results become evidence |
| Searcher improves its queries | The searcher may search again, with no record of why | Each new query is based on what the previous results lacked, and repeats are refused |
| Searcher stops within bounds | Search and page budgets | Also stops when evidence is sufficient or searches stop yielding accepted results, and records why it stopped |
| Searcher persists accepted evidence | Persistence happens after the searcher returns | The searcher persists its accepted sources and evidence before reporting |
| Short structured report | Summary, sufficiency flag, gaps | Per-item coverage with verified quotes, accepted/rejected sources, queries, stop reason |
| Supervisor finds essential gaps | Free-form judgment; a follow-up repeats the same task | Gaps are classified essential or non-essential against the acceptance criteria; essential gaps become targeted follow-up tasks |
| Claim-level faithfulness and citation support | One overall faithfulness score is the gate; citations are checked only for existence | Every factual claim is judged against its own cited evidence and this decides the gate; the overall score is kept as an informational signal |

Terminology: a **searcher** is the research sub-agent that researches one task (called the
"researcher" in the current system); the **supervisor** delegates tasks to searchers.

Affected workflow states (constitution, Delivery and Review Gates): planning (each task now
carries evidence points), researching (supervisor, searchers, follow-up tasks), research
readiness, verification and repair. Scoping, the plan approval step itself, human review and
publication approval are unchanged.

## Clarifications

### Session 2026-10-06

- Q: Where should each research task's list of evidence points come from? → A: The planner writes
  1–4 evidence points per task, each linked to the required answers it serves; they are shown with
  the plan for approval, and searchers cannot add new ones.
- Q: Should a follow-up for an essential gap be a new plan task or another round of the original
  task? → A: A new task with its own reworded question, not linked to the original task; coverage
  of the missing points is judged again over all tasks' reports.
- Q: Should the supervisor be allowed to start a follow-up task without a human approving it
  first? → A: Yes. Follow-ups run automatically within their caps and are recorded in the plan
  record and activity view; out-of-scope gaps use the existing replacement-plan approval.
- Q: When should the supervisor analyze searcher reports and create follow-up tasks? → A: After
  each round, once every task of that round has finished: round 1 is all approved plan tasks
  (however many batches the supervisor uses to delegate them), and each later round is the set of
  follow-up tasks created after the previous round. No follow-up is created mid-round.
- Q: Should the supervisor's model decide how and when to delegate tasks, or should the system
  delegate round 1 automatically? → A: The supervisor's model keeps deciding delegation in every
  round, including round 1, within its turn limit; the existing guarantee still researches any
  approved task it never delegated.
- Q: Should the claim-level check replace the existing overall faithfulness score or run alongside
  it? → A: The claim-level check is the faithfulness gate; the existing overall score still runs on
  the report that passed and is shown as an informational quality signal that never blocks.
- Q: Does a claim that its sources support only partly make the report fail? → A: Yes. A partly
  supported claim fails like an unsupported one, so every checked claim must be fully supported by
  its own citations; the threshold no longer gates the claim-level check.
- Q: When a claim is checked, which text of its cited source may the judge read? → A: Only the
  passages of that source recorded as the draft's evidence (what the writer was shown), never the
  rest of the stored page.
- Q: When a claim cites several sources, must every cited source support it, or is one enough? →
  A: Every cited source must support it; a cited source that does not makes the claim fail
  (refined by the next answer: each must support part of the sentence, together all of it).
- Q: Is a claim one sentence or one fact within a sentence? → A: One sentence. Each cited source of
  the sentence must support at least part of it, and the cited sources together must support all
  of it.
- Q: What evidence must a searcher attach when it marks an evidence point answered or partial? →
  A: The accepted source plus one short verbatim quote from that source's stored passages; a quote
  not found in the stored passages downgrades the point to missing.
- Q: Which passages should a searcher write its short report from? → A: The passages it read while
  searching; for any evidence point still missing or partial, it first searches the stored passages
  of its own accepted pages for that point before concluding (widened on 2026-10-07 to all pages
  already stored in the run).
- Q: How should the report's confidence level be defined, or should it be dropped? → A: Dropped;
  each evidence point's status, sources and verified quotes are enough.
- Q: How short must a searcher report be? → A: At most 400 words per report, each quote at most
  30 words, and at most 5 key findings (configurable defaults).
- Q: What happens when a searcher's report is invalid (bad structure, a missing evidence point, a
  rejected or unknown source)? → A: No rewrite; the system keeps the valid parts and records every
  invalid or missing part as a missing evidence point, so the supervisor's normal gap analysis
  decides whether an essential gap needs a follow-up task.

### Session 2026-10-07

- Q: When is the vector database queried to close a gap, and by whom? → A: Each searcher analyzes
  its own task's gaps first: for an uncovered evidence point it queries the passages of every page
  already stored in the run (its own and those of tasks that have finished), and only then spends
  remaining web searches on it. At the end of each round, the supervisor queries all of the run's
  stored pages for every essential point still missing before creating any follow-up task.
- Q: How should the query for an evidence point be formed? → A: The evidence point combined with its
  task's question, plus 2–3 alternative phrasings written by a model (including in the language of
  the sources); results of all phrasings are merged and de-duplicated.
- Q: How should stored passages be found and ranked for a lookup? → A: Hybrid: meaning-based search
  and exact-keyword search run together and their rankings are fused into one; no extra re-ranking
  model step.
- Q: How many passages does a lookup return, and is each widened with neighbouring passages? → A:
  The top 5 passages after fusion, without neighbouring passages; a point counts as not found when
  the model cannot give a verbatim quote from them that passes the quote check.
- Q: Should a searcher look up stored passages for every uncovered evidence point, or only essential
  ones? → A: Every uncovered evidence point, essential or not; the supervisor's end-of-round lookup
  and follow-up tasks remain for essential points only.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Searchers collect only evidence worth keeping (Priority: P1)

After the researcher approves a plan, each research task is handed to a searcher. The searcher
searches, judges every result against the task and the evidence the task needs, keeps the
results that help and discards the rest with a reason, refines its next query from what is still
missing, and stops when it has enough evidence or reaches its limits. Accepted pages and their
relevant passages are stored with full provenance before the searcher reports back.

**Why this priority**: Every later step (gap analysis, writing, verification) depends on the
evidence collected. Keeping irrelevant or low-quality pages wastes budget and produces weak or
unsupported reports.

**Independent Test**: Run one task against a controlled search source that returns a mix of
relevant, off-topic, duplicate and low-quality results. Confirm only relevant results are
accepted, each rejection has a reason, the second query differs from the first and targets what
was missing, the searcher stops within its limits with a recorded reason, and every accepted
source is stored and searchable with its provenance.

**Acceptance Scenarios**:

1. **Given** a search returns results, **When** the searcher evaluates them, **Then** each result
   is recorded as accepted or rejected, and each rejection carries one reason: off-topic, no
   usable content, duplicate, low credibility, or outside the task's scope.
2. **Given** the accepted evidence does not yet cover what the task needs, **When** the searcher
   searches again, **Then** the new query targets the uncovered points and is not a repeat of an
   earlier query of the same task.
3. **Given** a searcher is working, **When** its evidence covers every point the task needs, or its
   search budget is spent, or its page budget is full, or two consecutive searches produce no
   accepted result, or two consecutive searches fail (a failed result evaluation counts as a
   failed search), or its model ends the search on its own, **Then** it stops and records which
   of these happened.
4. **Given** a result is accepted, **When** the searcher finishes, **Then** the page's content and
   its relevant passages are durably stored and searchable, linked to the source address, the
   query that found it, the task, the run and the capture time.
5. **Given** a result is rejected, **When** the searcher finishes, **Then** it is not stored as
   evidence and can never be cited, but its address, query and rejection reason remain visible
   in the task's record.

---

### User Story 2 - Each searcher returns a short, structured report (Priority: P1)

When a searcher finishes, the supervisor receives a short report instead of raw pages: for each
point the task needed, whether it is answered, partly answered or missing and by which sources;
the key findings with source references; the accepted and rejected source counts; the queries
used; and why the searcher stopped.

**Why this priority**: The supervisor must judge coverage across tasks without reading every page.
A short, structured report keeps the supervisor's context small and its decisions auditable.

**Independent Test** (builds on User Story 1's searcher): Complete a task and inspect its report. Confirm every point the task needed
has a coverage status, every finding references at least one accepted source, the report stays
within its size limit, and it can be read back after the run resumes.

**Acceptance Scenarios**:

1. **Given** a finished searcher, **When** it reports, **Then** the report lists every evidence
   point of the task with a status (answered, partial, missing) and, for answered or partial
   points, the accepted source and a short verbatim quote from that source's stored passages.
2. **Given** a report quotes text that is not in the cited source's stored passages, **When** the
   report is received, **Then** that evidence point is downgraded to missing and the mismatch is
   recorded.
3. **Given** a finding in the report, **When** it is checked, **Then** it references at least one
   accepted source of this run, and no finding references a rejected or unknown source.
4. **Given** any searcher report, **When** it is stored, **Then** it stays within the configured
   size limit and is durably saved with the task.
5. **Given** a searcher that found no acceptable result, **When** it reports, **Then** every point
   is marked missing, the stop reason is stated, and the report says whether the cause was
   technical (search failure) or content (nothing relevant found).

---

### User Story 3 - The supervisor closes essential gaps with targeted follow-up tasks (Priority: P1)

When every task of a round has finished, the supervisor reads all searcher reports. A missing or
partial evidence point is essential when the approved plan links it to a required answer of the
brief. The supervisor first checks whether another task's findings already cover each essential
gap, then creates a new, narrow follow-up task for the gaps that remain, delegates it to a
searcher in the next round, and stops when every essential point is covered or the follow-up
limits are reached. Non-essential gaps are recorded as known limitations.

**Why this priority**: Gaps in essential points are the main reason reports fail verification.
Closing them during research is cheaper and safer than repairing a finished report.

**Independent Test** (builds on User Stories 1 and 2): Run a plan where one task leaves an
essential required answer missing and
another leaves a minor detail missing. Confirm exactly one follow-up task is created for the
essential gap, it names the missing point, it is researched by a searcher, the minor gap is
recorded as a limitation, and the follow-up limits are never exceeded.

**Acceptance Scenarios**:

1. **Given** all searcher reports of a round, **When** the supervisor analyzes them, **Then** each
   missing or partial evidence point is classified essential when the plan links it to at least
   one required answer of the brief, and non-essential otherwise; the classification names the
   linked required answers.
2. **Given** an essential gap, **When** follow-up allowance remains, **Then** the supervisor creates
   a new follow-up task with its own question and 1 to 4 evidence points that target the missing
   points, each linked to the required answers it serves (the task is not linked to the original
   task), and delegates it to a searcher.
3. **Given** a round has finished, **When** the supervisor analyzes the reports, **Then** each
   missing or partial essential point is judged against every task's report and then looked up in
   all of the run's stored pages, and is marked covered (naming the source and a verified quote)
   or still missing; only still-missing points can lead to a follow-up.
4. **Given** a follow-up task, **When** it is created, **Then** it stays within the approved plan's
   question and scope; a gap that needs research outside that scope is raised as a replacement
   plan proposal for human approval instead.
5. **Given** the follow-up allowance is spent or two consecutive follow-up rounds closed no
   essential gap, **When** essential gaps remain, **Then** research ends, the remaining gaps are
   recorded as limitations, and readiness reflects them.
6. **Given** no essential gap remains, **When** the supervisor analyzes the reports, **Then** it
   creates no follow-up task and research ends.

---

### User Story 4 - The final report is evaluated claim by claim (Priority: P1)

Before the researcher reviews the report, every factual claim in it is identified and checked:
whether the evidence supports it (faithfulness), whether the specific sources it cites support it
(citation support), and whether the report as a whole meets the acceptance criteria (required
answers, output requirements and the reviewer's requests). The researcher sees which claims
passed and which did not, and failing claims drive the repair.

**Why this priority**: A report-level score can pass while individual claims are unsupported or
cite the wrong source. Claim-level checks make every published claim traceable to evidence that
actually supports it.

**Independent Test**: Verify a report containing one supported claim with a correct citation, one
supported claim citing the wrong source, one unsupported claim, and one missing required answer.
Confirm the first passes, the second fails citation support, the third fails faithfulness, the
missing answer fails acceptance criteria, each result names the claim and its citations, and the
report does not pass.

**Acceptance Scenarios**:

1. **Given** a draft report, **When** it is evaluated, **Then** each sentence that states a fact is
   recorded as one claim with its citations and a verdict: supported, partly supported, unsupported or contradicted.
2. **Given** a claim sentence with citations, **When** citation support is checked, **Then** it
   passes only if each cited source supports part of the sentence and the cited sources together
   support all of it; a cited source that supports nothing, a part no cited source supports, or
   support found only in an uncited source is a citation failure naming the source or part.
3. **Given** a factual claim with no citation, **When** it is evaluated, **Then** it fails citation
   support unless it is explicitly labeled as analysis or uncertainty.
4. **Given** the acceptance criteria of the brief, **When** the report is evaluated, **Then** each
   criterion is recorded as met or not met with a justification.
5. **Given** the evaluation results, **When** any claim is partly supported, unsupported,
   contradicted or mis-cited, or any acceptance criterion is unmet, **Then** the report does not
   pass and the failing claims and criteria are handed to the existing repair cycle.
6. **Given** an evaluator cannot produce a trustworthy verdict, **When** evaluation runs, **Then**
   the outcome is an error, retried within its bound and then escalated, and the report is never
   rewritten because of an error.

---

### User Story 5 - The research process is observable and resumable (Priority: P2)

The researcher can follow, per task, the queries tried, accepted and rejected results, the stop
reason, follow-up tasks and their reasons, and the per-claim evaluation. If the application stops
mid-research, the run resumes without repeating completed searches or captures.

**Why this priority**: Trust and cost control. It builds on the P1 stories but is not needed to
produce a first correct report.

**Independent Test**: Interrupt a run after some tasks finished, resume it, and confirm finished
tasks are not searched again, follow-up decisions are preserved, and the activity view shows the
full history.

**Acceptance Scenarios**:

1. **Given** a run in progress, **When** the researcher opens its activity, **Then** each task
   shows its queries, accepted and rejected counts, stop reason, report status and any follow-up
   tasks with their gap.
2. **Given** an interrupted run, **When** it resumes, **Then** completed tasks and stored evidence
   are reused, and only unfinished tasks or follow-ups are researched.

---

### Edge Cases

- A search fails (timeout, provider error): it counts against the search budget, the error is
  recorded without credentials, and the searcher continues or stops by its limits.
- All results of every search are rejected: the task's report marks every point missing with a
  content stop reason; the supervisor treats essential points as gaps.
- The same page is accepted by two searchers running at the same time: it is stored once and both
  tasks reference that one capture.
- A page has no readable content but its snippet is relevant: it may be accepted as
  snippet-only evidence, recorded and shown as snippet-only (FR-012).
- An evidence point's answer is deep in an accepted page, outside the passages the searcher read:
  the per-point retrieval finds it, the point is reported answered with a quote, and no follow-up
  is created.
- The answer to a task's evidence point is on a page accepted by a task still running in parallel:
  the searcher cannot see it yet and does not wait; the supervisor's end-of-round lookup finds it
  and no follow-up is created.
- Ten tasks run three at a time: searchers only read pages already stored, and the supervisor's
  lookup runs after the whole round has finished, so no lookup blocks or waits for another task.
- A searcher report omits an evidence point and cites a rejected source for another: both points
  are recorded as missing, and the supervisor creates a follow-up only if either is essential.
- An evidence point asks for an exact number or name (for example a size limit or a version) that
  meaning-based search ranks low: the keyword ranking brings the exact-match passage into the
  results.
- The answer to an evidence point is split across two adjacent passages and the lookup returns only
  one: the point may be reported partial or missing; the end-of-round lookup or a follow-up can
  still close it.
- A searcher report exceeds the size limits: key findings are dropped first, then quotes are cut to
  the word limit (a cut quote must still match the stored passage verbatim); no evidence point's
  status, sources or quote is ever dropped.
- The supervisor proposes a follow-up for a gap already covered by another task's accepted
  evidence: the follow-up is refused and the gap is marked covered by that task.
- The supervisor reaches its turn limit with approved tasks never delegated: those tasks are still
  researched so no approved task is skipped.
- A claim cites a source id that does not exist in the run: it fails citation support.
- A claim is true according to its cited source's stored page but not according to the passages
  the writer was shown: it fails, and the repair cycle supplies the missing passage before the
  rewrite.
- The claim-level check passes a draft but the informational overall score is below the threshold:
  the draft still goes to human review, and both results are shown side by side.
- A sentence combines facts from two sources and cites both, each supporting one part: it passes;
  if it also cites a source that supports none of it, it fails naming that source.
- A sentence is an explicit assessment or statement of uncertainty: it is not judged as a factual
  claim but must be labeled as such.
- The reviewer's later "revise" or "extend" request adds new acceptance criteria: they are
  evaluated with the brief's criteria on the next draft.
- The supervisor delegates the approved tasks in several batches: no follow-up is created until the
  last approved task of round 1 has finished.
- The planner leaves a required answer of the brief linked to no evidence point: the plan is
  rejected and re-planned before it is shown for approval.
- A planner output has more than 4 evidence points for a task, or none: the plan is rejected and
  re-planned before it is shown for approval.

## Requirements *(mandatory)*

### Functional Requirements

#### Supervisor delegation

- **FR-001**: The supervisor MUST delegate only tasks of the approved plan and follow-up tasks it
  created under FR-016 to FR-019, and MUST refuse any unknown task.
- **FR-002**: The supervisor's model MUST decide which tasks of the current round to delegate and
  when, in every round including round 1. Each task MUST run in a separate searcher with an
  isolated working context, at most the configured number concurrently, and every task of the
  round MUST be researched at least once before the round ends: a task the model never delegated
  within its turn limit MUST be researched automatically.
- **FR-002a**: A second research attempt of the same task MAY be delegated only when the task's
  latest report records a technical failure; content gaps MUST be closed by follow-up tasks
  instead. Gap analysis MUST read the latest attempt's report of each task.
- **FR-003**: The supervisor's decision loop MUST be bounded by a configured maximum number of
  turns per round; together with the follow-up caps (FR-019) this bounds the whole research
  phase.
- **FR-003a**: The planner MUST give every research task 1 to 4 evidence points, each linked to the
  brief's required answers it serves (a point may link to none); the evidence points MUST be part
  of the plan shown for approval and of the approved plan's identity, so changing them requires a
  new approval. Every required answer of the brief MUST be linked to at least one evidence point of
  some task, or the plan is rejected before approval. Searchers MUST NOT add, remove or reword
  evidence points.

#### Searcher behavior

- **FR-004**: A searcher MUST search only through the approved search integration and MUST stay
  within its per-task search budget (at most 5 searches) and page budget.
- **FR-005**: A searcher MUST evaluate every new search result against its task and the task's
  evidence needs, and MUST record a decision for each result: accepted, or rejected with one reason
  (off-topic, no usable content, duplicate, low credibility, outside scope). If the evaluation of
  a search's results cannot be produced, none of that search's results is accepted and the search
  counts as a failed search for FR-007.
- **FR-006**: A searcher MUST base each query after the first on the evidence points not yet
  covered, and MUST NOT run a query that repeats an earlier query of the same task.
- **FR-007**: A searcher MUST stop when the first of these occurs and record it as the stop
  reason: all evidence points covered; search budget spent; page budget full; two consecutive
  searches with no accepted result; two consecutive search failures; the searcher's model ends
  its search on its own.
- **FR-008**: Only accepted results MAY count toward the page budget, become stored evidence or be
  cited in any report.

#### Persistence

- **FR-009**: Before reporting, a searcher MUST durably store each accepted page's cleaned content
  (or its snippet when no readable page exists) and add every passage of that content to the run's
  searchable evidence index, not only the passages selected while searching, each linked to the
  source address, title, query, task, run, capture time and its position in the page.
- **FR-010**: A page MUST be stored and indexed at most once per run, including when concurrent
  searchers accept it; every task that accepted it MUST reference that single capture.
- **FR-011**: Persistence MUST be safe to repeat: a resumed or retried task MUST NOT create
  duplicate stored pages or index entries.
- **FR-012**: The record of every evaluated result (accepted or rejected, reason, query, and
  whether an accepted result is snippet-only because its page had no readable content) MUST be
  stored with the task; snippet-only evidence MUST be marked as such wherever it is shown.

#### Searcher report

- **FR-013**: A searcher MUST return a structured report containing: the task, each evidence point
  with a status (answered, partial, missing) and its supporting accepted sources, key findings
  each referencing at least one accepted source, accepted and rejected counts, queries used, the
  stop reason, whether a failure was technical or content-related, and a report status (complete,
  corrected, unreadable). The report MUST NOT carry a self-assessed confidence level.
- **FR-013a**: Every answered or partial evidence point MUST carry, per supporting source, one short
  verbatim quote taken from that source's stored passages. Before the supervisor sees the report,
  the system MUST check each quote against the stored passages without a model call; a point
  whose quotes are all absent MUST be downgraded to missing, and the mismatch MUST be recorded.
- **FR-013b**: Each searcher MUST analyze its own task's gaps before reporting. For each evidence
  point not yet covered by the passages it read, whether essential or not, it MUST first retrieve passages for that point
  from every page of the run's active tasks already stored and indexed (its own accepted pages
  and those of active tasks that have finished; pages captured for a superseded plan are
  excluded); only if that finds nothing MAY it spend remaining web searches on the
  point. Retrieval MUST NOT count against the search budget, MUST only read stored pages (never
  block or wait for running tasks), and a retrieval failure MUST leave the point's status as it
  was and be recorded. The report MUST be written after the task's accepted pages are stored and
  indexed (FR-009).
- **FR-013c**: An invalid report MUST NOT be rewritten and MUST NOT fail the task. The system MUST
  keep its valid parts and record as missing every evidence point that is absent, unreadable, or
  supported only by rejected or unknown sources; findings that reference such sources MUST be
  dropped. Each correction MUST be recorded with the report. If no part of the report can be read,
  every evidence point is missing and the report status is unreadable.
- **FR-013d**: Every lookup of stored passages for an evidence point (FR-013b, FR-017a) MUST query
  with the evidence point combined with its task's question, plus 2 to 3 alternative phrasings
  written by a model, including at least one in the predominant language of the run's sources when
  it differs from the evidence point's language. Results of all phrasings MUST be merged and
  de-duplicated, and the phrasings used MUST be recorded with the lookup. If writing the
  alternative phrasings fails, the lookup MUST proceed with the combined query alone.
- **FR-013e**: Each lookup MUST rank stored passages by combining meaning-based similarity with
  exact keyword matching (so numbers, names and version strings are found), fusing the two
  rankings into one; no additional model-based re-ranking step is used. If keyword search is
  unavailable, the lookup MUST fall back to meaning-based ranking alone and record the fallback.
- **FR-013f**: Each lookup MUST return the 5 highest-ranked passages after fusion (configurable),
  without adding neighbouring passages. An evidence point MUST be treated as not found by a lookup
  when no verbatim quote from the returned passages passes the quote check (FR-013a); no relevance
  score threshold is used.
- **FR-014**: A searcher report MUST stay within configured size limits, by default at most 400
  words in total, at most 30 words per quote and at most 5 key findings, and MUST reference only
  accepted sources of the run.
- **FR-015**: Each report MUST be durably stored with its task and be the supervisor's only account
  of that task's research; to close gaps the supervisor MAY additionally retrieve the run's stored
  passages (FR-017a).

#### Gap analysis and follow-up tasks

- **FR-016**: Gap analysis MUST run once per round, after every task of the round has finished.
  Round 1 is all approved plan tasks, however many delegation batches the supervisor uses; each
  later round is the set of follow-up tasks created at the end of the previous round. The
  supervisor MUST NOT create a follow-up task before the current round has finished.
- **FR-016a**: In each analysis, every missing or partial evidence point MUST be classified
  essential when the approved plan links it to at least one required answer, and non-essential
  otherwise; the classification MUST name the linked required answers and MUST NOT depend on the
  supervisor's judgment.
- **FR-017**: For each essential gap within allowance, the supervisor MUST create a new follow-up
  task with its own question and 1 to 4 evidence points that target the missing points, each
  linked to the required answers it serves; the follow-up task is not linked to the task the gap
  came from. Non-essential gaps MUST be recorded as limitations.
- **FR-017a**: In each analysis, before creating any follow-up, the supervisor MUST judge each
  missing or partial essential point against all tasks' reports, and then retrieve passages for
  every point still uncovered from all stored pages of the run's active tasks. A point MAY be
  marked covered only
  with a short verbatim quote from a stored passage of an accepted source, checked without a model
  call as in FR-013a; the covering source, quote and justification MUST be stored. Points marked
  covered cannot receive a follow-up (FR-020).
- **FR-018**: Follow-up tasks MUST stay within the approved plan's question and scope and MUST run
  without a new human approval; each one MUST be added to the plan's record and shown in the
  activity view with the gaps that prompted it. A gap that needs research outside the approved
  scope MUST be raised as a replacement plan proposal requiring human approval.
- **FR-019**: Follow-up work MUST be bounded by a configured maximum number of follow-up tasks per
  required answer and per run, and research MUST end when two consecutive follow-up rounds close no
  essential gap. Round 1 never counts toward this rule; only follow-up rounds (round 2 onward) do.
- **FR-019a**: In each gap analysis the supervisor's model MUST get at most 4 turns to create
  follow-up tasks or propose a replan; when the turns are spent, the analysis ends with the
  follow-ups accepted so far and records that the turn limit was reached.
- **FR-020**: A follow-up for a gap already covered by another task's accepted evidence MUST be
  refused, and the gap MUST be marked covered.
- **FR-021**: Research readiness MUST be computed from the searcher reports and the gap analysis:
  ready when no essential gap remains, partial when only non-essential gaps or bounded-out essential
  gaps remain with usable evidence, insufficient when no usable evidence exists.

#### Final report evaluation

- **FR-022**: Evaluation MUST treat every sentence of the draft that states a fact as one claim, with
  the citations attached to that sentence; sentences explicitly labeled as analysis or uncertainty
  MUST be excluded from claim checks. A sentence that carries a citation presents sourced content
  and MUST always be treated as a claim, whatever the judge says about it.
- **FR-023**: Each claim MUST receive a faithfulness verdict (supported, partly supported,
  unsupported, contradicted) judged only against the passages recorded as the draft's evidence
  (the passages the writer was shown); the rest of a stored page MUST NOT be used, even when it
  contains the claim.
- **FR-024**: A claim MUST pass citation support only if the recorded passages of each of its cited
  sources support at least part of the sentence and, taken together, support all of it. A cited
  source that supports no part of the sentence, a part supported by none of the cited sources, an
  uncited factual sentence, a citation to an unknown source, or support found only in an uncited
  source MUST fail citation support; the result MUST name each cited source that supports nothing
  and each unsupported part.
- **FR-025**: Each acceptance criterion (required answers, output requirements, active reviewer
  requests) MUST be recorded as met or not met with a justification.
- **FR-026**: The claim-level check MUST be the only faithfulness gate. A draft MUST pass only when
  every checked claim is supported (a partly supported claim fails like an unsupported one) and
  passes citation support, and every acceptance criterion is met. The share of supported claims
  MUST still be recorded with the draft as its claim-level faithfulness result.
- **FR-026a**: The existing overall faithfulness score MUST still be computed, only for a draft that
  passed, and MUST be shown with the other quality signals as informational: a low score or a
  scoring failure MUST NOT fail the draft, re-enter repair or block human review, and the
  activity view MUST show it next to the claim-level result so the two can be compared.
- **FR-027**: Per-claim and per-criterion results MUST be durably stored with the draft version and
  MUST be handed to the existing repair cycle when the draft fails.
- **FR-028**: An evaluator failure MUST produce an error outcome, retried within its bound and then
  escalated to human review, and MUST NOT cause the report to be rewritten.

#### Safety and observability

- **FR-029**: Credentials MUST NOT appear in stored pages, result records, searcher reports, errors,
  logs or traces.
- **FR-030**: The activity view MUST show, per task, its queries, accepted and rejected counts, stop
  reason, report status and follow-up tasks with their gap; and, per draft, the claim verdicts and
  criterion results.
- **FR-031**: A resumed run MUST NOT repeat completed searches, evaluations, captures or reports, and
  MUST keep its follow-up decisions and limits.

### Key Entities

- **Research Task**: An approved plan question, or a follow-up task the supervisor created for
  essential gaps; has its own question, evidence points, status and the round it belongs to. A
  follow-up task records
  the gaps that prompted it but is not linked to another task.
- **Evidence Point**: One specific thing a task must find, written by the planner and approved with
  the plan; belongs to one task and links to zero or more of the brief's required answers, which
  makes it essential (linked) or non-essential (unlinked).
- **Result Evaluation**: One search result's decision for a task: address, query, accepted or
  rejected, and the rejection reason.
- **Accepted Source**: A page accepted by at least one task, captured once per run with its content
  (or snippet), provenance and searchable passages.
- **Searcher Report**: The structured summary of one task's research: evidence point statuses,
  findings with source references, verified quotes, counts, queries, stop reason, failure kind.
- **Research Round**: All tasks researched before one gap analysis: round 1 is the approved plan's
  tasks, each later round the follow-up tasks of the previous analysis; numbered, with its gap
  analysis and the follow-ups it produced.
- **Coverage Gap**: A missing or partial evidence point, classified essential or non-essential; an
  essential gap is either targeted by a follow-up task and later judged covered (by any task) or
  still missing, or recorded as a limitation.
- **Claim Evaluation**: One fact-stating sentence of a draft, its citations, its faithfulness verdict and
  its citation-support result.
- **Criterion Evaluation**: One acceptance criterion of a draft and whether it is met, with a
  justification.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of approved tasks are researched in every run, and no searcher, follow-up or
  supervisor loop exceeds its configured limit.
- **SC-002**: On a benchmark set, at least 90% of results judged off-topic or duplicate by a human
  are rejected by searchers, and at least 85% of results a human judges relevant are accepted.
- **SC-003**: Every stored evidence page traces to an accepted result, and no report cites a
  rejected or unknown source.
- **SC-004**: On a benchmark set with planted essential gaps, at least 80% are closed by follow-up
  tasks during research, and no follow-up task is created when no essential gap exists.
- **SC-005**: On a benchmark set with planted unsupported and mis-cited claims, at least 90% of them
  are flagged at claim level, and fewer than 5% of correctly supported claims are flagged.
- **SC-006**: Every claim in a published report has a recorded verdict and resolves to a supporting
  cited source; a reviewer can trace any claim to its passage in under one minute.
- **SC-007**: The share of reports that need a repair cycle to pass verification falls by at least
  25% on the benchmark set, without lowering the share that pass.
- **SC-008**: Searches per task fall by at least 20% on the benchmark set because searchers stop on
  sufficiency or diminishing returns, without lowering the share of answered evidence points.
- **SC-009**: No credential appears in any stored artifact, log, trace or user-visible error across
  the test suite and benchmark runs.

## Assumptions

- Existing scoping, plan approval (bound to the exact plan), human review, recovery, publication
  approval, run persistence and checkpoint resume are reused unchanged.
- The current limits stay as defaults: at most 5 searches per task, 3 concurrent searchers (at most
  5), 8 supervisor turns (now counted per round), 2 research attempts per task (the second only
  after a technical failure, FR-002a). New defaults,
  adjustable in planning: at most 5 accepted pages per task, 2 follow-up tasks per required
  answer, 4 per run, and searcher reports of at most 400 words, 30-word quotes and 5 key findings.
- "Evidence the draft was written from" is the evidence recorded for that draft version. How the
  writer selects that evidence is unchanged by this feature; this feature only requires that
  evaluation uses exactly that evidence.
- The faithfulness threshold keeps its current single configured value (strictly above 0.90) but
  applies only to the informational overall score (FR-026a): the claim-level gate requires every
  checked claim to be fully supported.
- Credibility judgments use general signals (primary or official source, recency, consistency with
  other sources); no source allow-list or block-list is introduced.
- A benchmark set of research questions with labeled results, planted gaps and planted claim errors
  exists or will be created for SC-002, SC-004, SC-005, SC-007 and SC-008.
