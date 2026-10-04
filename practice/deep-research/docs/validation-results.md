# Refactor validation results

Tests use fake models and tools with real VFS and SQLite storage. No paid provider calls were made.

- Ruff: passed (`python -m ruff check src tests`).
- Compilation: passed (`python -m compileall -q src`).
- Git diff whitespace: passed (`git diff --check`).
- Durability, retry and model factory tests: 14 passed.
- Worker, UI, model wiring, recovery and approval regression tests: 21 passed.
- Full suite: 315 passed, 2 skipped, 3 dependency deprecation warnings (40.53 seconds; the additional Windows regression was run separately).
- Windows VFS and path escape regression tests: 3 passed. The parallel worker test passed 5/5 repeated runs after normalizing extended-path prefixes.

Covered scenarios include approval before retrieval and replanning; VFS and Chroma retrieval
back to original evidence; targeted research and presentation repair; ERROR without rewriting;
the three-cycle limit; feedback batches and resume; acceptance with limitations and decline
while preserving FAIL; shared URL fetches; artifacts saved before checkpoint updates; saved
embeddings; stopped admission and completed sibling evidence; report, requirement and evidence
bindings; bounded retries and SDK retry ownership.

Two live integration tests are skipped without explicit opt-in and credentials. The new MCP
server has not been tested end-to-end with live Tavily, embedding providers and Chroma.
Exactly-once fetching is not guaranteed across crashes; each run requires one writer or recovery client.
