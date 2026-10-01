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
Each chat thread has one Langfuse trace (session ID = the durable `thread_id`) with a single
`research.thread` root that is opened on its first turn and ended when the thread is published,
needs review, or "New thread" is clicked, so its duration covers the whole thread. Every
submission adds one `research.turn` span under that root, which nests LangGraph node spans plus LangChain model generations. The trace
metadata correlates the durable `run_id`, per-submission `execution_id`, prompt version, role, model
attempt, structured-output validation, fallback, and safe error details. Export is flushed before the
execution completes; credential-shaped fields are redacted. Ragas evaluation uses
the deterministic local faithfulness baseline and does not publish a report.

A report that passes verification is scored (informational only) by the RAGAS judge
`INTELLIGENCE_FAITHFULNESS_MODEL`: Faithfulness, Context Precision (per research question, on the
ranked evidence the writer saw) and Response Relevancy (report body vs. the research brief, using
the vector index's embeddings). `INTELLIGENCE_EVAL_METRICS` picks the metrics (comma-separated;
default all, empty for none). Scores appear in the UI and as Langfuse trace scores, and are saved
to `verification/quality.<verification_id>.json`; the writer's ranking for each draft is saved to
`synthesis/retrieval.v<n>.json`.

Run the full local acceptance validation with `uv run pytest -q`; it should pass with only the live
MCP test skipped unless explicitly enabled. The controlled suite covers scope, todos-before-retrieval,
VFS evidence, synthesis, verification, recovery, resume, and approval binding.

The live MCP acceptance is opt-in only:

```text
$env:TAVILY_MCP_URL = "https://mcp.tavily.com/mcp/"
$env:TAVILY_API_KEY = "<your-api-key>"
$env:RUN_LIVE_MCP = "1"
uv run pytest -q tests/integration/test_live_mcp_opt_in.py
```

Keep the endpoint and API key in separate variables. The adapter adds `tavilyApiKey` only to the
in-memory transport URL; the credential is not included in request history, provenance, artifacts,
test output, or object representations.

The live Scope and Langfuse smoke test is also opt-in. Configure the blank variables documented in
`.env.example`, then run:

```text
$env:RUN_LIVE_LLM = "1"
uv run pytest -q tests/integration/test_scope_live_smoke.py
```

Confirm in Langfuse that the emitted trace is named `intelligence-collector`, its session ID equals
`live-smoke`, and the `research.turn` observation contains the Scope and Planning node spans with nested model
generations. The test never prints API keys.

Never commit `.env`, credentials, `.data/`, runtime checkpoints, or exported reports.
