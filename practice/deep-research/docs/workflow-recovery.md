# Workflow verification and recovery

Reviewer revisions now travel through `prepare_revision` before synthesis. This clears the
old verification, approval and quality scores. The writer sees the reviewed draft and active
requests; verification explicitly judges request satisfaction as well as factual support,
the research goal and per-question coverage. Missing or duplicated judge criteria are a
verifier error, not a failed report and never a pass.

Explicit English word limits are checked in Python. The count uses whitespace-separated
body words, includes heading text, and excludes citation markers, punctuation-only tokens
and the final Markdown `Sources` section. The latest standalone word limit supersedes
older standalone limits. Mixed or ambiguous instructions remain semantic requirements.
The draft is never truncated to pass a check, and failed drafts remain visible in Details.

## After a restart

Open **Recover an unfinished run**, select the run, and click **Restore checkpoint**.
For checkpoints created before the run registry existed, click **Find saved checkpoints** first.
This loads its original run/thread identity, not a new request. Reply normally to a
clarification or publication interrupt. Click **Continue unfinished run** for a pending
non-interrupted node.

At `NEEDS_REVIEW`, choose one of these explicit actions:

- `revise`: supply instructions; start a fresh bounded rewrite and verify again.
- `retry_report`: authorize one or two additional repair attempts.
- `research`: supply the missing-evidence instruction and research before rewriting.
- `rescope`: supply a changed research scope.
- `export_incomplete`: save a clearly labeled unverified artifact, never publish it.
- `cancel`: stop this run.

Only an explicit human approval bound to the exact passing draft can publish it.
Quality scores remain informational and do not bypass approval.

## Crash boundaries and compatibility

Saved research evidence restores its references and saved chunk IDs when its graph
checkpoint was not written. Research attempts retain their separate artifact directories.
Completed synthesis output is journaled by input state, writer prompt, model name and
configured report settings, so a crash before checkpointing can reuse it. Interrupt replies
are never cached. Publication rechecks authorization and does not replace identical bytes.

Provider calls still have an unavoidable crash window: a process dying before the result
is durably saved may repeat that in-flight call. This is not an exactly-once guarantee.
Recovery assumes a single writer per run; concurrent recovery clients are not supported.

Versionless checkpoints receive additive version-1 defaults. No migration invents evidence
or approval. Unknown future versions are rejected. Legacy ended clarification/review
checkpoints are moved through the existing graph-owned waiting edge on explicit restore.
Older research bundles without saved chunk IDs still restore their evidence, but their
missing vector references cannot be reconstructed automatically.
