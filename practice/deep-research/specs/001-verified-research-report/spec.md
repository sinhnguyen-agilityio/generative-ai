# Feature Specification: Verified Research Report

**Feature Branch**: `001-verified-research-report`

**Created**: 2026-09-24

**Status**: Draft

**Input**: User description: "Build a local deep research application that produces a verified,
evidence-backed technical report from a user's research request."

## Clarifications

### Session 2026-09-25

- Q: Should the structured output of `clarify_with_user` contain exactly the three fields
  `need_clarification`, `question`, and `verification`, with the detailed research scope created
  separately in `write_research_brief`? → A: Only the three clarification fields.
- Q: In what format should each assistant response appear in the chat transcript per turn? →
  A: One Markdown message per turn; structured JSON and workflow metadata appear only in Details.
- Q: On the default screen, how should agent, progress, and tool-call information appear so the
  chat stays as simple as ChatGPT? → A: Chat-first, with a status line and a Details panel that is
  collapsed by default.
- Q: While the workflow is running, how much detail about Scope, Planning, and tool calls should
  Streamlit show? → A: Show the current stage, the running step, and a short activity feed such as
  “Evaluating scope”, “Generating brief”, or “Calling write_todos”; hide sensitive tool arguments.
- Q: How should the indicator behave while the workflow is processing or waiting for the user to
  answer a clarification? → A: Animate the spinner only while the backend is executing; when the
  graph pauses for clarification, stop the spinner and show the static state “Waiting for your
  response”.
- Q: When should the activity feed update during Scope, Planning, and tool execution? → A: Update in
  real time with the lifecycle; create an item when a node or tool starts, then update that same
  item to success or failure when it finishes.
- Q: How should graph nodes and edges be organized to stay readable and maintainable? → A: Separate
  wiring (`build.py`), node logic (`nodes.py`), conditional routers (`routing.py`), and the
  tracing/checkpoint/lifecycle wrapper (`instrumentation.py`); node names come from a single shared
  source (FR-030a).
- Q: What should the Planner Agent receive and return? → A: It receives the research brief, uses an
  LLM to define a list of research questions (the output is only a list of questions), and then the
  Planning node calls `write_todos` to save them as the todo list (FR-037a).
- Q: How should the research agent and the Supervisor work? → A: The Supervisor assigns each todo to
  its own research sub-agent (at most 3 running in parallel, configurable); each sub-agent runs an
  LLM decision → Tavily MCP tool → compression loop, routed by the LLM's decisions (FR-008a,
  FR-009c).
- Q: When a sub-agent uses its whole search budget and the LLM judges the evidence insufficient,
  what status should the todo have? → A: Completed, with `sufficient: false` and unresolved gaps in
  the evidence bundle; Failed only when retrieval or compression produced nothing usable (FR-009d).
- Q: How should the Build-Verify-Fix loop verify and retry? → A: Verify with a deterministic,
  programmatic source-verification script over the report text; the agent may not finish normally
  until every check passes. Retries must be self-improving: reflect to derive the root cause,
  lessons, and a strategy, and accumulate lessons across attempts; the only other exit is
  `NEEDS_REVIEW` when the budget is exhausted or no progress is made (FR-016a, FR-017a, FR-017b,
  FR-019, FR-020). Superseded 2026-09-25 (added the acceptance check) and 2026-09-29 (single
  LLM-as-judge call replaces both stages); the retry/reflect/NEEDS_REVIEW behavior itself is
  unchanged.
- Q: How should repeated URLs be handled during research? → A: Track URLs by normalized identity;
  a sub-agent never re-shows or re-collects a page it already retrieved, and a run-level URL
  registry lets later tasks reference the first capture instead of storing and indexing the page
  again (FR-010c).
- Q: Should the Build/Fix prompt include every stored source, or a retrieved subset? → A: Retrieve
  a configurable top-K sources per research question (Gemini embeddings by default, Chroma,
  persisted locally), scoped per question so no question's sources can be crowded out by another's;
  Verification always reads the full, unfiltered evidence catalog independently, so narrowing the
  writer's prompt cannot change what a claim is checked against (FR-012a).
- Q: Which chat model should the live app use as primary, and what happens on failure? → A: Gemini
  (free tier) as primary, GPT as fallback, configured via environment variables and shared by every
  role through the one router each reused by default; an unset fallback keeps the app running with
  only the primary model. Superseded 2026-09-25 by per-role reasoning/standard tiers (FR-032b).

### Session 2026-09-24

- Q: When should the Scope Agent consider a request sufficiently clear and automatically transition
  to Planning? → A: Only when the objective, research boundaries, constraints, success criteria, and
  deliverable format are explicit, internally consistent, and testable; otherwise the agent helps
  the user narrow the scope.
- Q: What minimum structure and readiness checks must every todo satisfy before the workflow can
  enter Research? → A: Each todo contains a research question and status; Research starts only after
  a valid todo list is durably written, and each worker uses broad-to-narrow searches with a 2–3
  call budget for simple questions and an absolute 5-call limit for complex or unresolved questions.
- Q: Under what conditions may the Supervisor signal `ResearchComplete` and allow synthesis to
  begin? → A: Only when every todo is completed, intermediate artifacts are stored as files through
  the Virtual File System (VFS), and raw web research data is indexed in the vector store for later
  synthesis.
- Q: Which verification feedback taxonomy should deterministically select draft revision, additional
  research, or re-planning? → A: `DRAFT_DEFECT` triggers draft revision when adequate evidence exists;
  `EVIDENCE_GAP` triggers targeted additional research; and `PLAN_GAP` triggers re-planning affected
  todos.
- Q: Which file locations and publication actions should be permitted during research? → A: Each
  intermediate step writes a run-scoped VFS artifact proving that step occurred: Scope writes a brief
  summary, Planning writes the todo list, each Research task writes scraped website content, and
  Synthesis/Verification preserve the previous draft and latest report. Permanent output is written
  only after explicit human approval of the unchanged verified artifact.

- Q: What interaction pattern should the Scope Agent use for clarification and brief creation? → A:
  Use two sequential Scope nodes with separate structured LLM decisions: `clarify_with_user` reads
  the initial request and prior chat turns, returns either one focused clarification question or a
  clear decision, and pauses the same LangGraph thread when clarification is needed; once clear,
  `write_research_brief` makes a second structured LLM call to produce the brief and the graph
  automatically transitions to Planning without asking the user to approve the brief.

### Session 2026-09-25

- (Superseded 2026-09-30: planner acceptance criteria were removed; see FR-006a/FR-016c.)
- Q: How should the workflow know a report actually achieves the research brief's goal, beyond
  passing the deterministic citation/grounding checks? → A: The Planner also defines 1–5 acceptance
  criteria alongside the research questions (`TodoPlan.acceptance_criteria`); once the deterministic
  source-verification script passes, a separate LLM acceptance check judges the report against each
  criterion. Any unmet criterion is folded into the same verification result as a `DRAFT_DEFECT`, so
  it is fixed by the existing Build-Verify-Fix reflection loop rather than a new code path
  (FR-006a, FR-016b). Superseded 2026-09-29: the acceptance check is no longer a second stage
  gated on the deterministic script — both are replaced by one LLM-as-judge call (FR-016c).
- Q: Which roles should use a deeper-reasoning model tier versus a cheaper/faster standard tier? → A:
  Scope, Planner, and the acceptance check (role `verify`) use the reasoning tier; Researcher,
  research compression, Synthesizer, and Fixer use the standard tier. Each tier is configured with
  its own primary and optional fallback model via environment variables, replacing the single
  shared router every role previously used (FR-032b).

### Session 2026-09-29

- Q: Should Verify keep the two-stage deterministic-script-then-acceptance-check design (Session
  2026-09-25), or become a single LLM-as-judge call? → A: Single LLM-as-judge call
  (`agents/report_judge.py`, role `verify`), replacing both the deterministic
  `source_verification.py` script and the separate acceptance-criteria check as the graph's Verify
  gate. One call judges: each planner acceptance criterion, per-question coverage ("answers the
  question"), fair representation of the evidence, honest disclosure of uncertainty/evidence gaps,
  plain language free of unexplained abbreviations, and freedom from offensive/inappropriate
  content. Each unmet criterion carries its own judge-chosen issue type (`DRAFT_DEFECT` or
  `EVIDENCE_GAP`), so the existing rewrite-vs-research routing is preserved without a deterministic
  check driving it. The deterministic script still exists and still passes its own tests, but is no
  longer invoked by the graph — standalone/offline use only (FR-016a, FR-016c). Superseded the
  Session 2026-09-24 answer on verification taxonomy only insofar as *how* `DRAFT_DEFECT` /
  `EVIDENCE_GAP` are now decided (by the judge per criterion, not by a script); the taxonomy itself
  (FR-016, FR-017) is unchanged. Trade-off accepted: verification cost/latency now always includes
  one reasoning-tier LLM call (previously free/instant when the deterministic script alone
  sufficed), and citation-syntax enforcement (exact source ids, exact URLs, sentence-level citation
  presence) is judged by the LLM rather than checked with regex.
- Q: Should the Report Judge also check formatting/language conventions and do explicit fact
  checking, and if so how? → A: Two additions to the same single judge call (`build_report_judge_
  criteria`), not new agents or nodes:
  (1) An operator-configured **guideline checklist** (`Settings.report_guidelines`, env override
  `INTELLIGENCE_REPORT_GUIDELINES` as a JSON string array; default: Markdown format required,
  every sub-section must cite sources via hyperlinks or be erased/merged into the previous
  section, English only). Each guideline is appended to the criteria list verbatim and judged
  like any other criterion.
  (2) A fixed, non-configurable **fact-check criterion**: read the whole report start to finish
  and flag any factual inaccuracy, hallucination, or internal inconsistency against the evidence
  block already sent to the judge — not just claims that carry a citation. An unmet guideline or
  failed fact-check is a `DRAFT_DEFECT` like any other unmet criterion (the judge may still choose
  `EVIDENCE_GAP` per FR-016c's per-criterion rule when it decides the evidence itself is
  insufficient) and feeds the same Build-Verify-Fix fix/reflect loop. Considered and rejected:
  independent re-verification via new web searches per claim — real fact-checking against
  external reality, but adds search budget/latency to every Verify attempt; deferred as a
  possible future addition rather than built now (FR-016c, FR-016d).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Complete a Verified Research Run (Priority: P1)

A user submits a sufficiently clear technical research request and observes the run progress from
scope through planning, research, synthesis, and verification. The user then reviews a final report
whose factual claims can be traced to stored evidence.

**Why this priority**: Producing a trustworthy report is the feature's primary user value and the
minimum complete outcome.

**Independent Test**: Submit a clear request with available sources and confirm that the run creates
tasks before retrieval, completes research, verifies the draft, and presents a report with evidence
links without asking for approval of the research brief.

**Acceptance Scenarios**:

1. **Given** a valid request with a clear objective, constraints, success criteria, and deliverable
   format, **When** the user submits it, **Then** the run advances from Scoping to Planning without a
   research-brief approval prompt.
2. **Given** a scoped run in Planning, **When** planning completes, **Then** structured research tasks
   containing a research question and initial status have been durably written by `write_todos`
   before the first external source is retrieved.
3. **Given** planned research tasks, **When** research executes, **Then** each task has a visible
   status and its notes, excerpts, and evidence bundles are stored outside the chat transcript.
4. **Given** the supervisor has explicitly marked research ready for synthesis, **When** synthesis
   completes, **Then** verification checks scope coverage and factual-claim support against stored
   sources.
5. **Given** verification passes, **When** the user opens the result, **Then** the UI identifies the
   artifact as verified and shows the final report with navigable evidence references.
6. **Given** a simple research question, **When** a worker searches for evidence, **Then** it uses a
   budget of two or three search calls and stops early once it can answer confidently.
7. **Given** a complex question or unsuccessful searches, **When** a worker reaches its fifth search
   call, **Then** it stops searching and records the evidence found and any unresolved gap.
8. **Given** any todo is not completed or its required artifacts are not persisted and indexed,
   **When** the Supervisor assesses readiness, **Then** it MUST NOT signal `ResearchComplete` or
   permit synthesis to begin.
9. **Given** every todo is completed, **When** all intermediate artifact files are stored through the
   VFS and all raw web research data is indexed in the vector store, **Then** the Supervisor records
   `ResearchComplete` and synthesis may begin from those stored research assets.
10. **Given** a run advances through Scope, Planning, Research, and Synthesis/Verification, **When**
    each step completes, **Then** its required run-scoped VFS artifact exists: brief summary, todo
    list, scraped website content, and previous/latest report versions respectively.
11. **Given** a workflow execution is active, **When** Scope, Planning, or an approved tool runs,
    **Then** the chat-first UI shows a compact status line while agent and tool activity is available
    in a `Details` panel that is collapsed by default, without exposing credentials or sensitive
    tool arguments.
12. **Given** a graph node or tool call starts, **When** its lifecycle changes, **Then** one activity
    item appears immediately as running and that same item changes to success or failure when the
    operation ends, without creating misleading duplicate completion items.

---

### User Story 2 - Clarify an Ambiguous Request (Priority: P1)

A user begins with an underspecified request and answers focused follow-up questions until the
research objective is actionable.

**Why this priority**: Research begun from an ambiguous scope wastes effort and cannot be reliably
verified against user intent.

**Independent Test**: Submit a request missing its objective and output expectations, answer the
follow-up questions over multiple chat turns, and confirm that no planning or retrieval begins until
the required scope fields are clear.

**Acceptance Scenarios**:

1. **Given** a request with an unclear objective, constraint, success criterion, or deliverable
   format, **When** the `clarify_with_user` Scope node evaluates it, **Then** it returns exactly one
   focused follow-up question using only the three-field clarification response and the run remains
   in Scoping.
2. **Given** an ambiguous request, **When** the user answers only part of the missing information,
   **Then** the same LangGraph thread retains prior LangChain conversation turns and asks only for
   information still needed.
3. **Given** all required scope information has become clear, **When** the latest answer is accepted,
   **Then** the clear decision is recorded and the run automatically advances to
   `write_research_brief` and then Planning without brief approval.
4. **Given** a clear scope decision, **When** `write_research_brief` runs, **Then** it makes a
   separate structured LLM call and produces one detailed research question that consolidates the
   objective, constraints, success criteria, research boundaries, and deliverable format before the
   Planning transition.
5. **Given** malformed, empty, or unsafe input, **When** it is submitted, **Then** the input is
   rejected with an actionable error and no durable run state or external research is initiated.
6. **Given** a request whose objective is understandable but whose research scope is too broad,
   **When** the Scope Agent evaluates it, **Then** the agent proposes concrete narrowing options and
   keeps the run in Scoping until the user establishes explicit research boundaries.
7. **Given** the Scope graph pauses for clarification, **When** no backend work remains active,
   **Then** the UI stops all processing animation, displays “Waiting for your response”, and keeps the
   clarification question visible until the user replies.

---

### User Story 3 - Recover from Failed Verification (Priority: P1)

A user receives a verified result even when the first draft is incomplete or unsupported, provided
the problem can be repaired within the configured recovery limits.

**Why this priority**: Verification has value only when failures lead to controlled correction and
cannot be mislabeled as success.

**Independent Test**: Use evidence that causes the first draft to contain one unsupported claim,
confirm structured feedback selects an appropriate fix, and confirm the revised artifact is verified
again before it is shown as verified.

**Acceptance Scenarios**:

1. **Given** a draft that misses a scoped requirement or has an unsupported factual claim, **When**
   verification runs, **Then** the run records structured feedback identifying the affected section,
   claim or scope item, evidence gap, and required correction.
2. **Given** a verification failure, **When** a correction is selected, **Then** the workflow revises
   the draft, researches the specific gap, or re-plans affected tasks without discarding valid work.
3. **Given** a correction has completed, **When** the candidate artifact changes, **Then** the full
   verification gate runs again before any verified status is assigned.
4. **Given** all checks pass within the recovery limit, **When** verification completes, **Then** the
   latest artifact is marked verified and prior failed versions remain distinguishable in the audit
   history.
5. **Given** the recovery limit is exhausted with unresolved failures, **When** the final attempt
   ends, **Then** the run is marked `NEEDS_REVIEW`, the unresolved feedback is shown, and the artifact
   is never described as verified.
6. **Given** verification finds a `DRAFT_DEFECT`, **When** adequate stored evidence already exists,
   **Then** the workflow revises the draft without unnecessary retrieval.
7. **Given** verification finds an `EVIDENCE_GAP`, **When** a required claim lacks sufficient or
   consistent support, **Then** the workflow performs targeted additional research for that gap.
8. **Given** verification finds a `PLAN_GAP`, **When** existing todos do not cover or correctly
   decompose the scoped request, **Then** the workflow re-plans the affected todos before continuing.

---

### User Story 4 - Resume an Interrupted Run (Priority: P2)

A user reopens a run after the application or research process was interrupted and continues from
the last durable point without losing completed research.

**Why this priority**: Deep research may be long-running and must tolerate routine interruption
without costly or misleading repetition.

**Independent Test**: Interrupt a run after at least one research task and evidence bundle are
persisted, restart it, and confirm completed work remains available and execution resumes from the
last valid checkpoint.

**Acceptance Scenarios**:

1. **Given** a run interrupted after a durable checkpoint, **When** the user resumes it, **Then** its
   scope, task states, stored artifacts, evidence, attempt counters, and current stage are restored.
2. **Given** completed external research before interruption, **When** execution resumes, **Then**
   completed retrieval is not repeated unless its task was incomplete, invalidated, or explicitly
   selected for repair.
3. **Given** the newest checkpoint is missing or invalid, **When** resume is attempted, **Then** the
   UI reports a recoverable error, preserves existing artifacts, and offers the latest valid resume
   point or marks the run `NEEDS_REVIEW` if none exists.

---

### User Story 5 - Control Publication (Priority: P2)

A user reviews a verified artifact and explicitly approves or rejects publication to the permanent
output location.

**Why this priority**: Human control prevents drafts, unresolved results, or unwanted content from
being treated as an approved publication.

**Independent Test**: Reach a verified artifact, reject publication, and confirm no permanent output
is created; then approve the unchanged artifact and confirm only that exact version is published.

**Acceptance Scenarios**:

1. **Given** a verified artifact awaiting review, **When** the user rejects publication, **Then** no
   permanent output is created or replaced and the rejection is recorded.
2. **Given** a run that is unverified, failed, or `NEEDS_REVIEW`, **When** publication is requested,
   **Then** publication is denied with a reason and existing permanent output remains unchanged.
3. **Given** the user approves a verified artifact, **When** publication succeeds, **Then** the exact
   approved version is written to the permanent output location and the approval and publication
   outcome are recorded.
4. **Given** an approved artifact is edited before publication, **When** publication is attempted,
   **Then** prior verification and approval are invalidated and the changed artifact must be verified
   and approved again.
5. **Given** permanent output cannot be written, **When** publication is attempted, **Then** the user
   sees a publication failure, no partial artifact is represented as published, and retry does not
   require repeating completed research or verification of the unchanged artifact.

### Edge Cases

- The user abandons clarification or submits mutually contradictory scope answers.
- A planned task has no accessible sources, returns irrelevant sources, or repeatedly times out.
- An external research connector is unavailable, denies access, or returns malformed content.
- Multiple sources conflict, a source disappears, or an excerpt can no longer be traced to its
  origin.
- The supervisor receives partial worker results or a worker stops after persisting some artifacts.
- Research is declared ready while one or more required tasks are incomplete.
- The draft contains analysis or recommendations that are not factual claims but must be clearly
  distinguished from sourced facts.
- A correction introduces new claims or invalidates evidence links that previously passed.
- The process is interrupted during checkpoint creation or permanent publication.
- The user attempts to resume, approve, reject, or publish an unknown run or stale artifact version.
- The permanent output location already contains an artifact with the same requested identity.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a multi-turn local chat experience in which a user can submit
  and refine a technical research request.
- **FR-002**: The system MUST validate each user input for required content, supported size and
  format, and unsafe or malformed values before it changes durable state or initiates research.
- **FR-002a**: Every user request MUST be normalized (Unicode compatibility folding, confusable/
  homoglyph folding to their common-script equivalent, and removal of invisible/control characters)
  before guardrail checks, storage, or any model call, so lookalike characters cannot evade keyword
  filters, hide payloads later indexed into the vector store, or disguise sensitive patterns.
- **FR-002b**: Before the Scope Agent's clarification model runs, a pre-model content-safety
  guardrail MUST classify the user's latest message on the standard model tier and block it, without
  invoking the Scope Agent's model, when the message itself contains vulgar/obscene language or
  directly requests prohibited content; legitimate research about a sensitive real-world topic MUST
  remain allowed.
- **FR-003**: The Scope Agent MUST consider a request ready for Planning only when its objective,
  research boundaries, constraints, success criteria, and deliverable format are explicit,
  internally consistent, and testable.
- **FR-004**: When required scope information is unclear, the Scope Agent MUST ask focused follow-up
  questions, MUST preserve usable answers across turns, and MUST propose concrete ways to narrow a
  scope that is too broad.
- **FR-005**: Once scope is clear, the system MUST record the scoped request and automatically enter
  Planning without requiring approval of a research brief.
- **FR-006**: The Planner MUST call `write_todos` to durably create a non-empty list of structured
  research tasks before any external retrieval begins; a failed or invalid write MUST keep the run
  in Planning and expose an actionable error.
- **FR-006a**: Superseded 2026-09-30 — the Planner no longer defines acceptance criteria; Verify
  checks only the fact-check criterion and the operator-configured guidelines (FR-016c/FR-016d).
- **FR-007**: Each research task MUST contain a non-empty research question and a valid current
  status. Each newly written task MUST start in the planned status.
- **FR-008**: A Research Supervisor MUST assign planned tasks to research workers, track their
  outcomes, and record progress assessments.
- **FR-008a**: The Supervisor MUST delegate each planned todo to its own isolated research
  sub-agent and MAY run at most a configurable number of sub-agents concurrently (default 3,
  maximum 5). A failing sub-agent MUST become that task's recorded outcome without cancelling its
  siblings. Each sub-agent MUST have its own MCP client, so search budgets apply per task.
- **FR-008b**: Each research sub-agent MUST keep at most a configurable number of distinct source
  pages for its todo in total, across all of its searches (default 5, maximum 10) — the limit is per
  task, not per search call. Each search MUST request only the number of pages still available in
  that budget, and once the budget is full the sub-agent MUST stop searching and proceed to
  compressing its findings.
- **FR-008c**: A page is retrieved into a run once. While sub-agents search (including in
  parallel), the first to retrieve a URL owns it; another sub-agent whose search returns that URL
  MUST link to the existing capture instead of capturing or indexing it again, and the linked page
  MUST NOT count toward that sub-agent's source budget. This also applies to gap research and
  resumed runs for pages captured earlier in the same run.
- **FR-009c**: A research sub-agent MUST follow an LLM tool-calling loop: an LLM decision either
  requests `tavily_search` calls or gives a final note; a tool step executes the searches through
  the Tavily MCP connection; routing continues only while the LLM requests searches and budget
  remains; and a compression step makes a structured LLM call that condenses the raw results into
  cited findings (`ResearchFindings`: summary, sufficient, unresolved gaps). Search results MUST be
  treated as untrusted data.
- **FR-009d**: A research task MUST be marked completed when its sub-agent gathered at least one
  source and compressed findings, even if the findings are insufficient; insufficiency and
  unresolved gaps MUST be recorded in the evidence bundle for Verification. A task MUST be marked
  failed only when retrieval or compression produced nothing usable. MCP error messages MUST be
  redacted of credentials before being persisted.
- **FR-009**: Research workers MUST access external research sources only through approved MCP
  connections and MUST expose connection or retrieval failures as task outcomes.
- **FR-009a**: Before searching, a worker MUST read the task question and identify the specific
  information needed. It MUST start with a broad search, assess sufficiency and remaining gaps after
  every search call, and use narrower searches only to address identified gaps.
- **FR-009b**: A worker MUST stop searching as soon as it has enough evidence to answer confidently.
  It MUST use no more than three search calls for a simple question and no more than five for a
  complex question. Five search calls is an absolute per-task limit; reaching it MUST produce the
  best available outcome plus an explicit unresolved-gap record when adequate sources were not
  found.
- **FR-010**: Workers MUST persist research notes, source excerpts, and evidence bundles as
  intermediate files through the Virtual File System (VFS), outside conversation history, and
  associate each artifact with its run and task.
- **FR-010b**: The workflow MUST write a run-scoped VFS artifact after each completed stage: Scope
  writes a brief summary; Planning writes the todo list; each Research task writes the content scraped
  from its website sources; and Synthesis/Verification writes or preserves both the previous draft
  report and the latest report.
- **FR-010a**: Workers MUST preserve raw data retrieved from web sources with its source provenance
  and index that research data in the vector store for retrieval during synthesis.
- **FR-010c**: The workflow MUST track retrieved URLs by a normalized identity (case-insensitive
  scheme and host, no default port, fragment, trailing slash, or tracking parameters, sorted query)
  so each web page is captured and indexed at most once per run. Within one research sub-agent, a
  URL already retrieved MUST NOT be shown to the LLM again or collected twice. Across tasks, gap
  research, and resumed invocations, a run-scoped URL registry MUST make later evidence bundles
  reference the first capture (same source and artifact ids) instead of writing and indexing the
  page again.
- **FR-011**: The Supervisor MUST signal `ResearchComplete` before synthesis can begin and MAY signal
  it only when every todo has completed status, all required intermediate artifact files are
  durably stored through the VFS, and all raw web research data is indexed successfully.
- **FR-012**: The system MUST synthesize a versioned draft by retrieving from the indexed research
  data and stored evidence rather than relying on conversation history as the source of record.
- **FR-012a**: The Build/Fix prompt MUST select each research question's evidence by real
  embedding-based retrieval (not full-text dump) and MUST cap it at a configurable top-K sources
  per question, scoped to that question only, so one question's sources can never crowd out
  another's representation. Verification MUST read the complete, unfiltered evidence catalog
  independently of this selection, so narrowing the Build/Fix prompt MUST NOT change what a claim
  is checked against. A retrieval failure MUST NOT block Build; it MUST leave that question's
  evidence unfiltered instead.
- **FR-013**: The verifier MUST determine whether the draft addresses every required element of the
  scoped request.
- **FR-014**: The verifier MUST determine whether every factual claim intended for the final report
  is supported by one or more traceable stored sources.
- **FR-015**: Each factual claim in a verified artifact MUST link to stored evidence that identifies
  its source and preserves reviewable source context.
- **FR-016**: A failed verification MUST produce structured feedback with exactly one primary failure
  type: `DRAFT_DEFECT`, `EVIDENCE_GAP`, or `PLAN_GAP`; it MUST identify the affected scope item or
  claim, supporting details, and recommended repair path.
- **FR-016a**: Verification MUST be a single mandatory LLM-as-judge call (`agents/report_judge.py`,
  role `verify`) — there is no separate deterministic pre-check gating it. Superseded 2026-09-29;
  see FR-016c. The deterministic `source_verification.py` script MAY still be run standalone
  (non-zero exit code on failure) for manual/offline sanity-checking, but it is no longer part of
  the graph's Verify gate.
- **FR-016b**: Superseded 2026-09-29 by FR-016c — acceptance-criterion judgment is no longer a
  second, conditionally-run stage; it is folded into the single Report Judge call.
- **FR-016c**: Verification MUST be one LLM-as-judge call, with no search or scrape tools, that
  re-reads the report against the evidence the research step already gathered (it checks whether
  the Writer hallucinated while synthesizing, not whether the sources are true on the internet).
  It evaluates one fixed fact-check criterion — the report contains no factual inaccuracy,
  hallucination, or internal inconsistency when checked claim-by-claim against the gathered
  evidence (read as one continuous piece, introduction through conclusion — not only claims that
  carry a citation) — plus the operator-configured guidelines (FR-016d). Planner acceptance
  criteria, per-question coverage, fair-representation, uncertainty, plain-language and
  safe-content criteria were removed 2026-09-30. For each criterion the judge MUST return met/unmet, a
  justification (naming every specific problem found, not just the first, when unmet), and — when
  unmet — an issue type of its own choosing: `DRAFT_DEFECT` when the evidence supports the
  criterion but the writing does not show it, or `EVIDENCE_GAP` when the gathered evidence itself
  cannot satisfy the criterion no matter how the report is worded. Any unmet criterion turns the
  verification result into a failed `DRAFT_DEFECT` or `EVIDENCE_GAP` per that criterion's issue
  type, feeding the same Build-Verify-Fix fix/reflect-or-research loop as before.
- **FR-016d**: The system MUST also evaluate the report against an operator-configured guideline
  checklist (`Settings.report_guidelines`, default: Markdown format required; every sub-section
  must cite its sources via hyperlinks or be erased/folded into the previous section; English
  only) as additional criteria in the same FR-016c judge call. The checklist MUST be overridable
  via environment configuration (a JSON array of strings) without a code change, and each unmet
  guideline MUST be treated exactly like any other unmet FR-016c criterion.
- **FR-017**: The recovery workflow MUST revise the draft for `DRAFT_DEFECT`, research the specific
  gap for `EVIDENCE_GAP`, and re-plan affected tasks for `PLAN_GAP`.
- **FR-017a**: Each unresolved evidence gap MUST receive one targeted research round; gaps that remain
  afterwards MUST be disclosed under a `## Limitations` section rather than retried indefinitely.
- **FR-017b**: Retries in the Build-Verify-Fix loop MUST be self-improving rather than repeated
  generation: after a failed verification the agent MUST reflect on the failed checks to produce a
  root-cause diagnosis, reusable lessons, and a strategy; lessons MUST accumulate across attempts
  and be persisted; and every fix attempt MUST receive the failed checks, all lessons, the strategy,
  and the previous draft. When a fix reproduces the same failures (same failure fingerprint), the
  next reflection MUST be told the previous strategy failed and change approach; two consecutive
  no-progress verifications MUST end the loop in `NEEDS_REVIEW`.
- **FR-018**: Every repaired candidate artifact MUST pass verification again; prior verification
  MUST NOT carry over to changed artifact content.
- **FR-019**: Every retry and recovery path MUST have a configured finite limit and visible attempt
  count.
- **FR-020**: When a recovery limit is exhausted without a passing verification result, the system
  MUST mark the run `NEEDS_REVIEW`, preserve unresolved feedback, and MUST NOT label the artifact as
  verified.
- **FR-021**: The system MUST maintain durable checkpoints sufficient to restore the run's scoped
  request, current stage, task states, stored-artifact references, supervisor assessments, draft
  versions, verification history, recovery counters, and publication state.
- **FR-022**: A resumed run MUST continue from the latest valid checkpoint and MUST NOT repeat
  completed external work unless that work is incomplete, invalidated, or selected for repair.
- **FR-023**: The UI MUST be chat-first: its primary transcript contains only user and assistant
  messages, matching a simple conversational layout. While work is active, a compact status line
  MUST show the current stage and action. Agent identity, chronological activity, tool calls,
  task-level research progress, verification outcome, and recovery details MUST remain available in
  a `Details` panel that is collapsed by default. The latest reviewable report remains available in
  the conversation. The UI MUST NOT expose credentials or sensitive tool arguments.
- **FR-023a**: The UI MUST animate a processing indicator only while backend execution is active. A
  graph paused for clarification MUST show a non-animated “Waiting for your response” state with the
  outstanding question visible and MUST NOT imply that processing continues.
- **FR-023b**: UI progress MUST update during execution from graph-node and approved-tool lifecycle
  events. Each operation MUST have one stable activity item that transitions from running to success
  or failure and retains chronological ordering.
- **FR-023c**: Each completed conversational turn MUST add at most one assistant message to the
  primary transcript. That message MUST be plain Markdown containing only the focused clarification
  question, a concise research-start acknowledgement, or the reviewable report appropriate to that
  turn. Structured JSON, internal state, agent metadata, and tool metadata MUST NOT appear in the
  transcript and MAY appear only in the collapsed `Details` panel.
- **FR-023d**: The UI MUST offer a "New thread" action that starts a new run and thread identity
  (fresh `run_id`/`thread_id`, cleared transcript/activity/pipeline state) without discarding cached
  model, tracer, or graph connections. Starting a new thread MUST NOT alter any previous thread's
  durable VFS artifacts, checkpoints, or publication record.
- **FR-023e**: Alongside the transcript, the UI MUST render one row per pipeline agent (Scope
  Agent, Planner, Researcher, Synthesizer, Verifier), each combining that agent's content on the
  left with its own running/complete/failed status indicator on the right of that same row,
  derived from that agent's own lifecycle activity. A row MUST stay hidden until that agent has
  actually started processing — the pipeline MUST NOT list every agent as pending before a
  request exists. While an agent is running, its row MUST show a processing indicator; once it
  completes, its row MUST show that agent's key output for the current run: the Scope Agent's
  research brief, the Planner's research questions as a list, and, for the Researcher, every
  completed research question's retrieved source links.
- **FR-023f**: These rows MUST update live, node by node, while a request is being processed —
  driven by the graph's own execution stream (not only redrawn once after the entire run
  completes) — so the operator can watch each agent progress from running to complete during a
  single request instead of seeing one static summary at the end.
- **FR-024**: The system MUST distinguish at least the states Scoping, Planning, Researching,
  Synthesizing, Verifying, Awaiting Approval, Published, Failed, and `NEEDS_REVIEW`.
- **FR-025**: Only a verified artifact MAY enter Awaiting Approval, and explicit human approval of
  its exact version MUST be recorded before publication.
- **FR-026**: Rejecting publication MUST leave the permanent output location unchanged and record
  the rejection without representing the artifact as published.
- **FR-026a**: Rejection MAY include free-text reviewer feedback. When present, the system MUST
  classify it as either a formatting-only request (`format`: only the presentation of the existing
  report changes) or a scope change (`scope`: what the report should cover changes), and route the
  run accordingly. Rejection without feedback, or without a classifier configured, MUST default to
  the `format` path rather than discarding the existing research plan.
- **FR-026b**: A `format` rejection MUST NOT re-plan or re-research and MUST NOT change any claim,
  number or citation of the report; a `scope` rejection MUST re-enter scope clarification with the
  feedback as a new user turn (the scope agent MAY ask a clarifying question), rewrite the research
  brief, and produce a fresh set of research questions, restarting the attempt counters. Either path re-enters the normal
  Build-Verify-Fix loop and MUST return to Awaiting Approval only once the revised or replanned
  artifact fully passes verification again — it MUST NOT publish or re-offer the prior, rejected
  artifact.
- **FR-027**: The system MUST deny publication for an unverified, changed-after-approval, failed, or
  `NEEDS_REVIEW` artifact and provide an actionable reason.
- **FR-028**: Publication MUST place only the exact approved artifact version in the permanent output
  location and MUST record success or failure without exposing partial output as published.
- **FR-028a**: Research, planning, synthesis, and verification MUST NOT write to the permanent output
  location. Publication MUST be the only action that writes there, and it MUST require explicit human
  approval tied to the unchanged verified artifact version.
- **FR-029**: Errors MUST identify the affected stage or task, preserve completed durable work, and
  tell the user whether the run can retry, resume, or requires review.
- **FR-030**: Workflow stage transitions, bounded recovery attempts, resumable checkpoints, and
  approval gates MUST be governed by the project's required workflow orchestration mechanism.
- **FR-030a**: Workflow nodes MUST be identified by a single shared set of stable node names used by
  graph wiring, UI progress, and tracing. Every conditional transition MUST be a named routing
  decision over typed workflow state, and a missing or empty verification result MUST NOT route to
  Awaiting Approval.
- **FR-031**: Semantic decisions made by Scope, Planning, Supervision, Research, Synthesis,
  Verification, and Fixer roles MUST use a real LLM call in live mode with a prompt template and a
  typed, validated output. Deterministic fakes MAY be selected explicitly for offline tests but MUST
  NOT silently replace the live model.
- **FR-032**: The model router MUST centralize primary/fallback models, reasoning settings, timeouts,
  retry limits, primary model name, errors, and structured decisions in
  run traces (a fallback model's run is captured by the LangChain callback trace) without recording secrets.
- **FR-032a**: The opt-in live LLM/MCP smoke tests MUST reuse the reasoning profile's
  `INTELLIGENCE_REASONING_MODEL`/`INTELLIGENCE_REASONING_PROVIDER` rather than a standalone
  model config, so there is exactly one place to configure the reasoning tier.
- **FR-032b**: The live application MUST route every role to one of exactly two model profiles by
  role name — there is no per-role or per-scope model config: a reasoning profile for `scope`,
  `planner`, and `verify` (the acceptance-criteria check), and a standard profile for `researcher`,
  `research_compressor`, `synthesizer`, and `fixer`. Each
  profile MUST configure its own primary model, optional fallback model, provider, and reasoning
  kwargs from environment variables (`INTELLIGENCE_REASONING_*`, `INTELLIGENCE_STANDARD_*`) so the
  operator can change either tier without a code change. An unset fallback MUST leave that
  profile's router with only its primary model configured.
- **FR-033**: The `clarify_with_user` LLM output MUST contain exactly three fields:
  `need_clarification` (boolean), `question` (string), and `verification` (string). When clarification
  is needed, `question` MUST contain exactly one focused question and `verification` MUST be empty.
  When scope is clear, `question` MUST be empty and `verification` MUST contain a concise
  acknowledgement. A clear result automatically transitions to brief generation; clarification
  pauses the same LangGraph thread for the next Streamlit turn.
- **FR-033a**: Scope clarification MUST use two sequential decisions: a `clarify_with_user` node
  evaluates the initial request plus prior LangChain conversation and either pauses with one focused
  question or emits a clear three-field decision; then `write_research_brief` performs a separate
  structured LLM call to derive the detailed scope from authoritative user turns and create the
  brief. The graph MUST invoke `write_research_brief` only after a clear decision and MUST transition
  to Planning automatically after the brief is recorded.
- **FR-033b**: `write_research_brief` MUST return a dedicated validated `ResearchBrief` containing
  exactly one non-empty, detailed research question. The question MUST consolidate the objective,
  boundaries, explicitly stated constraints, success criteria, and deliverable format from the
  conversation without inferring or inventing user preferences or constraints.
- **FR-033c**: A clarification graph invocation MUST append exactly one assistant question and end.
  A clear decision MUST append exactly one concise assistant verification message that acknowledges
  sufficient information, summarizes the understood request, and confirms research will begin. The
  graph MUST then route automatically to brief generation without asking the user to approve it.
  Assistant verification is explanatory context and MUST NOT become authoritative evidence of a
  user preference or constraint unless that information also appears in a user message.
- **FR-034**: All LLM-powered roles MUST use LangChain chat models initialized through
  `langchain.chat_models.init_chat_model` via the central model factory/router. LangGraph remains the
  workflow orchestrator; the Deep Agents SDK is prohibited.
- **FR-035**: Role prompts MUST be `ChatPromptTemplate` instances (no manual version labels; the
  trace records the full rendered prompt) using
  `MessagesPlaceholder` for prior turns where applicable. User text and retrieved evidence MUST be
  passed as data variables, not interpolated into system instructions.
- **FR-036**: Every model output that can change workflow state MUST be parsed into its Pydantic schema
  before routing. Invalid output MUST produce a recorded error or bounded retry and MUST NOT silently
  become `SCOPE_CLEAR` or `VERIFIED`.
- **FR-037**: The Planning node MUST consume the validated `ResearchBrief`, create at least one
  planned research todo, and successfully persist it through `write_todos` before any edge can enter
  Researching or invoke external retrieval. Brief or todo persistence failure MUST end the current
  invocation visibly without enabling retrieval.
- **FR-037a**: The Planner Agent MUST decompose the `ResearchBrief` with a structured LLM
  call whose output is a list of research questions (`TodoPlan.questions`, 1–5 items). Each
  question MUST be standalone, non-blank, and not duplicate another (case-insensitive). The LLM
  MUST NOT set todo IDs or statuses; the system converts each question into a `planned` todo and
  persists the todo list (through `write_todos`). Invalid planner output MUST NOT write a todo
  list.
- **FR-038**: Every live workflow execution MUST emit one complete Langfuse trace before it may be
  treated as successfully observed. Each execution trace MUST be grouped under the LangGraph
  `thread_id`, correlated with `run_id`, and contain a root execution observation, child graph-node
  spans, and nested LangChain model-generation observations including model identity,
  attempts, fallback status, validated result or safe error metadata, and final stage. Missing or
  invalid tracing configuration MUST fail a live execution before any model call. Credentials and
  secret values MUST never appear in trace inputs, outputs, metadata, errors, or logs.
- **FR-038a**: Every Langfuse observation input and output MUST be human-readable text. Structured
  model observations MUST contain the complete rendered instructions and conversation supplied to
  the model plus the validated response serialized as JSON text. Hashes, IDs, and
  dispatch statuses MAY remain metadata but MUST NOT replace semantic input or output content.
- **FR-039**: Once a draft passes verification (FR-016c), the system MUST score the whole report's
  RAGAS Faithfulness against the excerpt
  text of every source the report actually cites, using an LLM judge, and record the resulting
  score (or, on judge failure, the error) as a durable per-run artifact. This score is
  observational: a missing, low, or failed score MUST NOT alter verification's pass/fail
  determination, the workflow stage, or publication eligibility (`can_request_approval` remains
  keyed only to `VerificationResult.passed`).
- **FR-039a**: The faithfulness score, when present, MUST be displayed to the user alongside the
  report.

### Key Entities *(include if feature involves data)*

- **Research Run**: One execution for a user's request; includes identity, current stage, status,
  timestamps, checkpoint reference, and terminal outcome.
- **Scoped Request**: The agreed objective, constraints, success criteria, deliverable format, and
  clarification history that define what verification must assess.
- **Research Task**: A planned unit of investigation whose required todo fields are a research
  question and status; assignment, search-call count, stored artifacts, unresolved gaps, and outcome
  are recorded as the task progresses.
- **Progress Assessment**: A supervisor record of task coverage, gaps, blockers, and readiness for
  synthesis.
- **Research Artifact**: A persisted note, source excerpt, or other intermediate work product tied to
  a run and task, stored as a file through the VFS outside chat history.
- **Stage Artifact**: A run-scoped VFS file proving a completed stage, including the scope brief
  summary, planning todo list, per-task scraped website content, and previous/latest report versions.
- **Indexed Research Data**: Raw web-retrieved content with source provenance, associated with its
  run and task and indexed in the vector store for synthesis retrieval.
- **Source Record**: The identity, location, retrieval context, and relevant metadata of an external
  source accessed through an approved research connection.
- **Evidence Bundle**: Reviewable source excerpts and provenance supporting one or more report claims.
- **Report Artifact**: A versioned draft or final technical report with scope coverage and links from
  factual claims to evidence.
- **Verification Result**: The outcome, checks performed, structured failures, and affected artifact
  version for a verification attempt, including one primary `DRAFT_DEFECT`, `EVIDENCE_GAP`, or
  `PLAN_GAP` type when verification fails.
- **Checkpoint**: Durable workflow state needed to resume a run without depending on conversation
  history.
- **Approval Decision**: A human approval or rejection tied to an exact verified artifact version.
- **Publication Record**: The requested destination, approved artifact version, outcome, timestamp,
  and any failure details for a publication attempt.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In acceptance testing, 100% of clear valid requests advance from Scoping to Planning
  without asking the user to approve a research brief.
- **SC-002**: In 100% of tested runs, structured research tasks exist before the first external
  retrieval attempt, every task has a non-empty research question and planned status, and
  `write_todos` completed successfully.
- **SC-003**: In 100% of artifacts labeled verified, every scoped requirement has an explicit
  verification result and every factual claim has at least one reviewable evidence link.
- **SC-004**: A user can determine the run's current stage, completed and remaining research work,
  latest verification outcome, and report availability within 10 seconds using the compact status
  line and, when needed, the collapsed-by-default `Details` panel.
- **SC-004a**: In 100% of tested chat turns, the primary transcript adds no more than one assistant
  Markdown message and contains no structured decision JSON, internal state payload, or tool
  metadata.
- **SC-005**: In the defined failed-verification acceptance test, the system records the defect,
  performs the targeted repair, and re-verifies the changed artifact before assigning verified
  status.
- **SC-006**: In 100% of recovery-limit exhaustion tests, the run ends as `NEEDS_REVIEW` and no UI or
  output describes its artifact as verified.
- **SC-007**: In interruption tests at each major workflow stage, 100% of runs resume from a valid
  durable point with completed artifacts intact and without repeating completed retrieval work.
- **SC-008**: In 100% of rejection and unauthorized-publication tests, the permanent output remains
  unchanged and the user receives a clear reason.
- **SC-009**: In 100% of successful publication tests, the published content exactly matches the
  artifact version that the human approved.
- **SC-010**: At least 90% of representative first-time users can submit a request, answer any scope
  questions, inspect evidence, and make a publication decision without assistance.
- **SC-011**: In 100% of worker tests, no task exceeds five search calls; simple tasks use no more
  than three, and tasks stop earlier when their question can be answered confidently.
- **SC-012**: In 100% of tested runs, synthesis remains blocked until every todo is completed, all
  required intermediate files are stored through the VFS, and all raw web research data is indexed;
  synthesis retrieves its research context from those durable assets rather than chat history.
- **SC-013**: In 100% of stage-completion tests, the required stage artifact is present in the run's
  VFS, and in 100% of rejected or unapproved publication tests the permanent output is unchanged.

## Assumptions

- The first release serves one local user at a time; multi-user accounts, roles, and remote hosting
  are outside this feature's scope.
- The permanent output location is a user-configured local destination. Publication to third-party
  content platforms is outside scope.
- A technical report is the only required deliverable type for this feature, though its requested
  structure and format may vary by scoped request.
- The user is responsible for having lawful access to configured external research sources.
- Source quality may vary; verification establishes traceability and scope coverage, not universal
  truth or endorsement of every source.
- Retry and recovery limits are configurable, but every configured value is finite and visible in
  run status.
- Conversation history supports the chat experience but is not authoritative storage for workflow
  state, research artifacts, evidence, or report versions.
- The project-mandated local Python, Streamlit, LangGraph, `write_todos`, and MCP constraints are
  accepted dependencies; this specification defines observable behavior rather than their internal
  design.
