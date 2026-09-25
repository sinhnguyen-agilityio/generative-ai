# Quickstart Validation Guide

This guide validates behavior with deterministic fake models, fake MCP responses, a temporary VFS,
and an isolated local checkpoint/vector index. No live external source is required for the core
acceptance demonstrations.

## Prerequisites

1. Python 3.11+ and the project virtual environment.
2. Install project dependencies and configure local VFS, checkpoint, vector-index, and output paths.
3. Configure fake MCP fixtures and deterministic model responses for tests. Configure Langfuse only
   when tracing is being demonstrated; configure the fixed Ragas evaluation dataset for evaluation.

## Validation commands

```text
pytest tests/unit tests/contract
pytest tests/integration/test_end_to_end_controlled.py
pytest tests/integration/test_resume.py tests/integration/test_resume_stages.py
pytest tests/integration/test_model_fallback.py
pytest tests/integration/test_bvf_routes.py
pytest tests/integration/test_approval_binding.py
pytest tests/integration/test_guardrails.py
pytest tests/integration/test_langchain_scope.py tests/integration/test_scope_checkpoint.py
pytest tests/integration/test_langfuse_tracing.py
pytest tests/evaluation/test_ragas_faithfulness.py
pytest tests/ui
streamlit run app.py
```

## Required demonstrations

- **End-to-end**: submit a bounded request; observe Scope → Planning → Researching →
  `ResearchComplete` → Synthesis → Verification → Awaiting Approval. Assert `todos.v1.json` exists
  before any MCP call, raw captures are indexed, claims link to evidence, and the UI shows progress.
- **Ambiguous scope**: submit a broad request; assert narrowing questions and no `write_todos` or MCP
  call until boundaries are explicit; verify multiple turns append one question each, then a separate
  brief call automatically enters Planning and persists todos without brief approval.
- **Scope model failures**: force malformed output, primary failure with fallback success, and total
  primary/fallback failure; assert only validated outputs route and no failure is treated as clear.
- **Langfuse hierarchy**: enable Langfuse, run one clear request and one clarification turn, then
  verify each submission has a root execution trace grouped under the same thread/session, graph-node
  child spans, nested model generations, prompt/model/attempt metadata, final stage, and no secrets.
- **Resume**: interrupt after a research artifact and checkpoint; restart with the same `thread_id`;
  assert artifacts and task status restore and completed MCP work is not repeated.
- **Model fallback**: force a primary model timeout; assert bounded retry, fallback selection,
  Langfuse fallback event, and successful continuation or visible terminal failure.
- **BVF routes**: inject each typed verifier issue and assert `DRAFT_DEFECT` revises only the draft,
  `EVIDENCE_GAP` adds targeted evidence, and `PLAN_GAP` adds/revises affected todos. Assert each
  repair changes the progress fingerprint; repeat no-op repairs until `NEEDS_REVIEW`.
- **Approval binding**: approve a verified hash, mutate the candidate, and assert publication is
  denied; reject approval and assert permanent output is unchanged; approve unchanged content and
  assert atomic publication exactly matches the approved hash.
- **Guardrails**: submit prompt injection and disallowed offensive/violent requests; assert safe
  rejection, no checkpoint/artifact/tool mutation, and actionable UI feedback.
- **Faithfulness**: run the fixed Ragas dataset against stored evidence contexts and record the
  repeatable score and run configuration in evaluation artifacts.
