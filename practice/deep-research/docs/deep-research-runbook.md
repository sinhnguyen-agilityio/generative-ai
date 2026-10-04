# Deep Research Local Runbook

Run the controlled acceptance suite without credentials:

```text
uv sync --extra test
uv run pytest -q
uv run ruff check src tests
```

Artifacts are written under `.data/vfs/<run_id>/`; checkpoints and vector index are local files.
Permanent output is under `exports/` and is written only after approval bound to the verified report
hash. Langfuse tracing is optional: with `INTELLIGENCE_LANGFUSE_ENABLED=true` and keys configured,
each research run is one Langfuse trace (ID derived from `run_id`, session ID = `thread_id`), and
every graph invocation (new request, plan or report approval, recovery) adds one
`research.execution` with the graph nodes, research sub-agents, MCP calls and LLM generations
beneath it. A missing or rejected key disables tracing with a warning; research runs unchanged.
Credentials and long payloads are masked before export. See `docs/observability.md`.

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

Confirm in Langfuse that the emitted trace is named `deep-research`, its session ID equals
`live-smoke`, and its `research.execution` observation contains the `research_graph` run with the
Scope and Planning node observations and their nested model generations. The test never prints
API keys.

Never commit `.env`, credentials, `.data/`, runtime checkpoints, or exported reports.
