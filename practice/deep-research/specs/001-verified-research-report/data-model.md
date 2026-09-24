# Data Model

All records include `schema_version`, `run_id`, `created_at`, and stable IDs. Artifact content is
content-addressed by SHA-256 and written atomically under the run's VFS root.

## Graph State

`ResearchGraphState` contains:

- `run_id`, `thread_id`, `stage`, `status`, `input_turns`
- `scoped_request`, `scope_questions`, `scope_complete`
- `todo_ids`, `todo_statuses`, `research_complete_requested`
- `task_outcomes`, `artifact_refs`, `evidence_refs`, `index_refs`
- `progress_assessment`, `draft_versions`, `verification_result`
- `recovery_attempt`, `recovery_budget`, `progress_fingerprint`
- `approval_decision`, `publication_record`, `last_error`

Only graph nodes may change workflow fields; UI reads snapshots.

## Core Records

### ScopedRequest

`objective`, `research_boundaries`, `constraints`, `success_criteria`, `deliverable_format`,
`clarification_turn_ids`, `scope_hash`. Planning requires all fields explicit, consistent, and
testable.

### ResearchTodo

`todo_id`, `run_id`, `research_question`, `status` (`planned`, `assigned`, `in_progress`,
`completed`, `failed`). A valid todo has a non-empty question and starts as `planned`.

### SourceRecord and ResearchArtifact

Source records retain URL/locator, retrieval time, MCP server/tool, request parameters, and source
identity. Artifacts retain kind (`brief_summary`, `todo_list`, `raw_capture`, `note`, `excerpt`,
`evidence_bundle`, `progress_assessment`, `draft`, `verification`), task ID, source IDs, VFS path,
hash, and version.

### IndexedChunk

`chunk_id`, artifact/source IDs, text span, embedding/index version, metadata filters, and hash.
Synthesis retrieval must return chunk IDs and source provenance.

### VerificationResult

`verification_id`, artifact version/hash, `passed`, checks, claim links, and exactly one issue type
(`DRAFT_DEFECT`, `EVIDENCE_GAP`, `PLAN_GAP`) when failed, with affected IDs and repair details.

### ApprovalDecision and PublicationRecord

Approval binds actor, timestamp, artifact hash, verification ID, and decision. Publication records
destination, approved hash, atomic-write outcome, and error details. Any content hash change
invalidates approval.

## Lifecycle

`Scoping → Planning → Researching → Synthesizing → Verifying → AwaitingApproval → Published`.
Failures may enter `Failed` or `NEEDS_REVIEW`; BVF repair routes return to Researching, Synthesizing,
or Planning. `ResearchComplete` ends Researching only.

## VFS Layout

```text
<vfs_root>/<run_id>/
  scope/brief_summary.v1.json
  planning/todos.v1.json
  research/<todo_id>/raw/<artifact_id>.json
  research/<todo_id>/notes/<artifact_id>.md
  research/<todo_id>/evidence/<artifact_id>.json
  research/<todo_id>/index/<artifact_id>.json
  supervisor/progress/<artifact_id>.json
  reports/draft.v<n>.md
  verification/result.v<n>.json
  checkpoints/<checkpoint_id>.json
  approvals/<approval_id>.json
```

Permanent output is outside this root and writable only by the publisher.
