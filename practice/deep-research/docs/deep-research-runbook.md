# Deep Research Local Runbook

Run the controlled acceptance suite without credentials:

```text
uv sync --extra test
uv run pytest -q
uv run ruff check src tests
```

Artifacts are written under `.data/vfs/<run_id>/`; checkpoints and vector index are local files.
Permanent output is under `exports/` and is written only after approval bound to the verified report
hash. Langfuse is optional and must be configured with environment variables. Ragas evaluation uses
the deterministic local faithfulness baseline and does not publish a report.

The live MCP acceptance is opt-in only:

```text
$env:RUN_LIVE_MCP = "1"
uv run pytest -q tests/integration/test_live_mcp_opt_in.py
```

Never commit `.env`, credentials, `.data/`, runtime checkpoints, or exported reports.
