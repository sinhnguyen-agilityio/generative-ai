---
name: graph-test-writer
description: Writes offline pytest tests for deep-research agents and graph nodes. Use when asked to add or extend tests.
---

Write tests for the named agent or node under `tests/`.

- Never call a live LLM or search. Use the scripted models in `tests/fakes.py`; mirror the style of `tests/test_researcher.py` and `tests/test_verifier.py` (plain functions, descriptive names that state behavior).
- Reuse existing fakes; add a new one to `tests/fakes.py` only if none fits, and keep that file free of pytest imports.
- Cover the normal path, one routing/edge case, and a failure path.
- Run `uv run pytest <file>` and fix until it passes. Report what the tests cover.
