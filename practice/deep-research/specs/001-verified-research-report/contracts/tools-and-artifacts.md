# Tool, Artifact, Verification, and Publication Contracts

## `write_todos`

Input: non-empty array of `{todo_id, research_question, status: planned}`. Output: durable todo
list artifact reference and normalized task IDs. It rejects empty questions, duplicate IDs, invalid
status, and cross-run writes. A failed call blocks Research.

## `think_tool`

Input: task statuses, artifact/index references, gaps, and budget counters. Output:
`ProgressAssessment` with coverage, blockers, next action, and `research_complete_eligible`. It is
advisory; only the validated completion node may transition the graph.

## Evidence bundle

Contains source IDs, excerpts, locator, retrieval timestamp, MCP provenance, task ID, claim targets,
and content hash. Unsupported claims cannot reference a bundle.

## Verification

Output is `{passed, artifact_hash, checks, issues[]}`. Failed output contains exactly one primary
issue type: `DRAFT_DEFECT`, `EVIDENCE_GAP`, or `PLAN_GAP`, plus affected IDs and repair details.

## Publication

Input is `{run_id, artifact_hash, verification_id, approval_id, destination}`. The publisher checks
hash equality, verified status, approval binding, and destination policy, then performs an atomic
write. Rejection, stale hash, or failed write leaves permanent output unchanged.
