<!--
Sync Impact Report
- Version change: template (unratified) -> 1.0.0
- Modified principles:
  - Template Principle 1 -> I. Validated Human Input
  - Template Principle 2 -> II. Explicit, Resumable Workflow
  - Template Principle 3 -> III. Durable Research Artifacts
  - Template Principle 4 -> IV. Evidence-Linked Research
  - Template Principle 5 -> V. Verified, Human-Approved Publication
- Added sections:
  - Architecture and Integration Constraints
  - Delivery and Review Gates
- Removed sections: None
- Follow-up TODOs: None
-->
# Deep Research Constitution

## Core Principles

### I. Validated Human Input
All user-supplied input MUST be validated before it can affect workflow state, tool
arguments, persisted artifacts, or rendered output. Validation MUST enforce the expected
type, shape, size, and allowed values; reject unsafe or malformed input with an actionable
error; and preserve the user's original intent without silently inventing missing facts.
Validation failures MUST NOT initiate research or mutate durable state. This boundary is
required because chat input is untrusted and can influence external operations.

### II. Explicit, Resumable Workflow
LangGraph MUST own research workflow transitions. Each transition MUST have explicit entry
criteria, outputs, and failure behavior. Recovery and retry loops MUST have configured bounds
and MUST terminate in a visible success, failure, or human-intervention state. Material
workflow state MUST be checkpointed so execution can resume after interruption without
repeating completed external work or losing provenance. Ad hoc control flow outside the
graph MUST NOT bypass these transition guarantees.

### III. Durable Research Artifacts
Intermediate research artifacts MUST be persisted in dedicated storage outside conversation
history. Stored artifacts MUST use stable identifiers and retain sufficient metadata to
associate them with the originating run, workflow step, source, and creation time. Chat
messages MAY summarize or reference artifacts but MUST NOT be their system of record.
Resumed runs MUST load authoritative state from artifact and checkpoint storage.

### IV. Evidence-Linked Research
External research sources MUST be accessed through approved MCP integrations. Every factual
claim intended for a final artifact MUST link to one or more stored evidence records that
identify the source and preserve enough source context for review. Unsupported claims MUST
be removed, explicitly labeled as analysis or uncertainty, or sent back for additional
research. Source access and evidence storage MUST preserve provenance across retries and
resumed execution.

### V. Verified, Human-Approved Publication
Publication is a gated workflow transition, not an automatic consequence of drafting. A
candidate final artifact MUST pass verification for claim-to-evidence coverage, source
traceability, required structure, and unresolved errors or uncertainties. After verification,
an authorized human MUST explicitly approve the exact artifact version to be published.
Edits after approval invalidate that approval and require verification and approval again.
No failed, unverified, or unapproved artifact may be represented as final or published.

## Architecture and Integration Constraints

- The application MUST use Python and provide its conversational user experience through
  Streamlit.
- LangGraph MUST be the authoritative workflow state machine and its checkpoint mechanism
  MUST support resumable execution.
- Conversation history, workflow checkpoints, research artifacts, evidence records, and final
  artifacts MUST have distinct responsibilities and storage boundaries.
- External research access MUST occur through MCP clients with explicit configuration,
  timeouts, bounded retries, and observable failure states.
- Secrets and credentials MUST NOT be written to conversation history, research artifacts,
  evidence records, logs, or published output.
- Persisted claim and evidence records MUST remain addressable by stable identifiers so a
  reviewer can traverse from final claim to stored evidence.

## Delivery and Review Gates

Every feature specification and implementation plan MUST identify affected workflow states,
validation boundaries, persistence behavior, MCP dependencies, evidence requirements, and
human approval points. Automated tests MUST cover rejected input, legal and illegal state
transitions, retry exhaustion, checkpoint resume behavior, artifact persistence, evidence
link integrity, verification failure, and publication without approval. Integration tests
MUST exercise MCP failure handling and checkpoint restoration without depending on
conversation history as durable storage.

Reviewers MUST reject changes that introduce unbounded loops, bypass LangGraph transitions,
store authoritative artifacts only in chat history, create factual claims without evidence
links, or permit publishing without verification and explicit human approval. Any exception
MUST be documented as a time-bounded constitutional amendment; implementation convenience is
not sufficient justification.

## Governance

This constitution supersedes conflicting development practices and feature-level documents.
Amendments MUST be proposed as a documented change, state the rationale and migration impact,
receive human approval, and update the Sync Impact Report before adoption. Compliance MUST be
reviewed during specification, planning, implementation review, and before release.

Constitution versions follow semantic versioning: MAJOR for incompatible principle removals
or redefinitions, MINOR for new principles or materially expanded obligations, and PATCH for
clarifications that do not change obligations. The Last Amended date MUST change whenever the
version changes. Deviations discovered after release MUST block publication where safety,
evidence integrity, resumability, or approval guarantees are affected and MUST produce a
remediation plan for all other violations.

**Version**: 1.0.0 | **Ratified**: 2026-09-24 | **Last Amended**: 2026-09-24
