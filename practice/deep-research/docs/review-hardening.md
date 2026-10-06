# Workflow review hardening

The workflow requested after the original September specification supersedes its
LLM-only verification and no-research-in-BVF decisions. The current contract is:

- Verify combines semantic goal/request/evidence checks with deterministic
  citation-ID, Sources-URL and requested-format checks. Structural checks do not
  prove factual truth. Verifier errors/truncation never become a passing verdict.
- Missing core evidence reopens only affected todos with a new attempt. Automatic
  replanning has been removed. Re-scoping archives old
  evidence but excludes it from the active report catalog; source IDs remain stable.
- Research assessment returns READY, PARTIAL or INSUFFICIENT. Optional gaps do not
  block a supported answer. Only INSUFFICIENT gets one automatic targeted attempt
  before synthesis. If still insufficient, or PARTIAL, synthesize an honest incomplete
  answer, verify it, then end without approval/publication. Report repair
  has a separate configured allowance (default three); stalled repairs can stop early.
- Only the "Approve and publish" button (`{"approved": true}`) authorizes publication; a typed
  reply is always a change request. Publication remains bound to the verified artifact hash
  and verification ID.
- Failed publication invalidates approval/verification and enters the recovery
  interrupt. Legacy ended approval checkpoints enter this same safe recovery path.
  Retry/revise must verify and obtain new approval; export is explicitly incomplete.
- Successful research outcomes are saved as each worker finishes, before indexing.
  An indexing failure can replay the saved outcome without paying for research again;
  sibling persistence continues. Returned graph results remain in plan order.
- Legacy `.v1` artifact paths remain compatible latest aliases; overwritten content
  is retained in a content-addressed immutable sibling revision.
- Missing Tavily configuration is visibly scope/plan-only, not a finished report.

Per-call output caps remain provider token ceilings, not report word-count criteria.
Reasoning defaults are low effort with a 4096-token provider ceiling; role caps still
apply. No fixed dollar guarantee is claimed and no paid API calls are required by tests.

Windows long-path failures still require a reproducing platform trace before a path
layout migration. Prefer a short configured data root on affected installations.
