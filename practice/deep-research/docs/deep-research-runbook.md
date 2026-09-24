# Deep Research Local Runbook

Run the controlled acceptance suite without credentials:

```text
uv sync --extra test
uv run pytest -q
uv run ruff check src tests
```

Artifacts are written under `.data/vfs/<run_id>/`; checkpoints and vector index are local files.
Permanent output is under `exports/` and is written only after approval bound to the verified report
hash. Live app executions require Langfuse credentials configured through environment variables.
Each submission creates one `research.execution` trace, groups executions by Langfuse session using
the durable `thread_id`, and nests LangGraph node spans plus LangChain model generations. The trace
metadata correlates the durable `run_id`, per-submission `execution_id`, prompt version, role, model
attempt, structured-output validation, fallback, and safe error details. Export is flushed before the
execution completes; credential-shaped fields are redacted. Ragas evaluation uses
the deterministic local faithfulness baseline and does not publish a report.

Run the full local acceptance validation with `uv run pytest -q`; it should pass with only the live
MCP test skipped unless explicitly enabled. The controlled suite covers scope, todos-before-retrieval,
VFS evidence, synthesis, verification, recovery, resume, and approval binding.

The live MCP acceptance is opt-in only:

```text
$env:RUN_LIVE_MCP = "1"
uv run pytest -q tests/integration/test_live_mcp_opt_in.py
```

The live Scope and Langfuse smoke test is also opt-in. Configure the blank variables documented in
`.env.example`, then run:

```text
$env:RUN_LIVE_LLM = "1"
uv run pytest -q tests/integration/test_scope_live_smoke.py
```

Confirm in Langfuse that the emitted trace is named `intelligence-collector`, its session ID equals
`live-smoke`, and the root observation contains the Scope and Planning node spans with nested model
generations. The test never prints API keys.

Never commit `.env`, credentials, `.data/`, runtime checkpoints, or exported reports.
