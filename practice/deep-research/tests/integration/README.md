# Controlled MVP Run

Run `uv run pytest -q tests/integration/test_end_to_end_controlled.py tests/integration/test_mvp_artifacts.py`.
The fixtures are deterministic and assert todos are persisted before MCP retrieval, raw evidence is
indexed, stage artifacts are written, and synthesis/verification completes without live network.
