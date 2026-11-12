# deep-research

Multi-agent research app: LangGraph graph, Streamlit UI, Chroma store, Langfuse tracing, RAGAS evaluation.

## Commands
- Tests: `uv run pytest`
- Lint/format: `uvx ruff check --fix` and `uvx ruff format` (a hook runs both on each edited `.py`)
- Run the app: use the `run-deep-research` skill
- Trace inspection: `python scripts/fetch_langfuse_trace.py [--trace-id ID | --thread-id ID --hours N]`

## Layout
- `src/deep_research/agents/` graph nodes; `schemas/` Pydantic structured outputs; `prompts.py` all prompts
- `src/deep_research/utils/routing.py` graph routing; `orchestrator.py` builds the graph
- `tests/fakes.py` scripted models and search, so tests run offline

## Conventions
- Never read or grep `.env` files (a hook blocks it).
- After implementing, update spec.md, plan.md and tasks.md to match the code. Write docs in English.
- Delete obsolete code in the same change; no speculative additions.
- Commit messages: one-line subject only.
