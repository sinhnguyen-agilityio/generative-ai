# Feature Specification: Verified Research Report

**Feature Branch**: `001-verified-research-report`

**Created**: 2026-09-24

**Status**: Draft

**Input**: User description: "Build a local deep research application that produces a verified,
evidence-backed technical report from a user's research request."

## Clarifications

### Session 2026-09-25

- Q: Trong lúc workflow đang chạy, Streamlit nên hiển thị mức độ chi tiết nào về Scope,
  Planning và tool calls? → A: Hiển thị stage hiện tại, bước đang chạy và activity feed ngắn như
  “Evaluating scope”, “Generating brief”, hoặc “Calling write_todos”; ẩn tool arguments nhạy cảm.
- Q: Indicator nên hoạt động thế nào khi workflow đang xử lý hoặc đang chờ người dùng trả lời
  clarification? → A: Chỉ animate spinner khi backend đang thực thi; khi graph pause để chờ
  clarification, dừng spinner và hiển thị trạng thái tĩnh “Waiting for your response”.
- Q: Activity feed nên cập nhật vào thời điểm nào trong quá trình Scope, Planning và tool
  execution? → A: Cập nhật realtime theo lifecycle; tạo item khi node hoặc tool bắt đầu rồi cập nhật
  chính item đó thành success hoặc failure khi kết thúc.

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
    **Then** the UI shows the current stage, a concise current-action label, and a chronological
    activity item without exposing credentials or sensitive tool arguments.
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
   focused follow-up question and the run remains in Scoping.
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
- **FR-007**: Each research task MUST contain a non-empty research question and a valid current
  status. Each newly written task MUST start in the planned status.
- **FR-008**: A Research Supervisor MUST assign planned tasks to research workers, track their
  outcomes, and record progress assessments.
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
- **FR-011**: The Supervisor MUST signal `ResearchComplete` before synthesis can begin and MAY signal
  it only when every todo has completed status, all required intermediate artifact files are
  durably stored through the VFS, and all raw web research data is indexed successfully.
- **FR-012**: The system MUST synthesize a versioned draft by retrieving from the indexed research
  data and stored evidence rather than relying on conversation history as the source of record.
- **FR-013**: The verifier MUST determine whether the draft addresses every required element of the
  scoped request.
- **FR-014**: The verifier MUST determine whether every factual claim intended for the final report
  is supported by one or more traceable stored sources.
- **FR-015**: Each factual claim in a verified artifact MUST link to stored evidence that identifies
  its source and preserves reviewable source context.
- **FR-016**: A failed verification MUST produce structured feedback with exactly one primary failure
  type: `DRAFT_DEFECT`, `EVIDENCE_GAP`, or `PLAN_GAP`; it MUST identify the affected scope item or
  claim, supporting details, and recommended repair path.
- **FR-017**: The recovery workflow MUST revise the draft for `DRAFT_DEFECT`, research the specific
  gap for `EVIDENCE_GAP`, and re-plan affected tasks for `PLAN_GAP`.
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
- **FR-023**: The UI MUST show the current stage, concise current action, chronological activity feed,
  task-level research progress, latest verification outcome, recovery status when applicable, and
  the latest reviewable report. Scope, Planning, and approved tool calls MUST emit user-readable
  activity items such as “Evaluating scope”, “Generating brief”, and “Calling write_todos”; the UI
  MUST NOT expose credentials or sensitive tool arguments.
- **FR-023a**: The UI MUST animate a processing indicator only while backend execution is active. A
  graph paused for clarification MUST show a non-animated “Waiting for your response” state with the
  outstanding question visible and MUST NOT imply that processing continues.
- **FR-023b**: UI progress MUST update during execution from graph-node and approved-tool lifecycle
  events. Each operation MUST have one stable activity item that transitions from running to success
  or failure and retains chronological ordering.
- **FR-024**: The system MUST distinguish at least the states Scoping, Planning, Researching,
  Synthesizing, Verifying, Awaiting Approval, Published, Failed, and `NEEDS_REVIEW`.
- **FR-025**: Only a verified artifact MAY enter Awaiting Approval, and explicit human approval of
  its exact version MUST be recorded before publication.
- **FR-026**: Rejecting publication MUST leave the permanent output location unchanged and record
  the rejection without representing the artifact as published.
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
- **FR-031**: Semantic decisions made by Scope, Planning, Supervision, Research, Synthesis,
  Verification, and Fixer roles MUST use a real LLM call in live mode with a versioned prompt and a
  typed, validated output. Deterministic fakes MAY be selected explicitly for offline tests but MUST
  NOT silently replace the live model.
- **FR-032**: The model router MUST centralize primary/fallback models, reasoning settings, timeouts,
  retry limits, prompt versions, selected model, fallback events, errors, and structured decisions in
  run traces without recording secrets.
- **FR-033**: Scope Agent LLM output MUST be either `NEEDS_CLARIFICATION` with exactly one focused
  question or `SCOPE_CLEAR` with objective, constraints, success criteria, research boundaries, and
  deliverable format. `SCOPE_CLEAR` automatically transitions to Planning; clarification pauses the
  same LangGraph thread for the next Streamlit turn.
- **FR-033a**: Scope clarification MUST use two sequential decisions: a `clarify_with_user` node
  evaluates the initial request plus prior LangChain conversation and either pauses with one focused
  question or emits a clear decision; then `write_research_brief` performs a separate structured LLM
  call to create the brief. The graph MUST invoke `write_research_brief` only after a clear decision
  and MUST transition to Planning automatically after the brief is recorded.
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
- **FR-035**: Role prompts MUST be versioned `ChatPromptTemplate` instances using
  `MessagesPlaceholder` for prior turns where applicable. User text and retrieved evidence MUST be
  passed as data variables, not interpolated into system instructions.
- **FR-036**: Every model output that can change workflow state MUST be parsed into its Pydantic schema
  before routing. Invalid output MUST produce a recorded error or bounded retry and MUST NOT silently
  become `SCOPE_CLEAR` or `VERIFIED`.
- **FR-037**: The Planning node MUST consume the validated `ResearchBrief`, create at least one
  planned research todo, and successfully persist it through `write_todos` before any edge can enter
  Researching or invoke external retrieval. Brief or todo persistence failure MUST end the current
  invocation visibly without enabling retrieval.
- **FR-038**: Every live workflow execution MUST emit one complete Langfuse trace before it may be
  treated as successfully observed. Each execution trace MUST be grouped under the LangGraph
  `thread_id`, correlated with `run_id`, and contain a root execution observation, child graph-node
  spans, and nested LangChain model-generation observations including model identity, prompt version,
  attempts, fallback status, validated result or safe error metadata, and final stage. Missing or
  invalid tracing configuration MUST fail a live execution before any model call. Credentials and
  secret values MUST never appear in trace inputs, outputs, metadata, errors, or logs.
- **FR-038a**: Every Langfuse observation input and output MUST be human-readable text. Structured
  model observations MUST contain the complete rendered instructions and conversation supplied to
  the model plus the validated response serialized as JSON text. Prompt versions, hashes, IDs, and
  dispatch statuses MAY remain metadata but MUST NOT replace semantic input or output content.

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
  latest verification outcome, and report availability from a single UI view within 10 seconds.
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
