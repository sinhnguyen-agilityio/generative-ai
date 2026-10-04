# Observability: Langfuse traces of a research run

This describes how a research run appears in Langfuse and why. The implementation is
`src/intelligence_collector/observability/tracing.py` (Langfuse Python SDK **4.15**, which is
built on OpenTelemetry). The tests that prove the topology are
`tests/integration/test_trace_topology.py`, `tests/integration/test_langfuse_tracing.py`,
`tests/ui/test_streamlit_trace_boundary.py` and `tests/contract/test_observability_contract.py`.

## Trace boundary

| Langfuse concept | Maps to | How it is derived |
|---|---|---|
| Trace | One research run (`run_id`) | `create_trace_id(seed="research-run:<run_id>")`: the same ID in every process |
| Session | The LangGraph thread (`thread_id`) | `propagate_attributes(session_id=thread_id)` |
| `research.execution` | One graph invocation | Opened around `graph.stream`/`graph.invoke`, closed when it returns |

A run is not one invocation. It pauses for a human at least twice (plan approval and
report approval), and each answer resumes the graph in a later Streamlit request, possibly in
another process after a restart. Keeping one observation open across those pauses would need
in-memory state that a restart loses: the parent would never be exported and every later
observation would be orphaned. So each invocation is its own `research.execution`, closed when
it returns, and all of them share the run's trace ID. Because that ID is derived from `run_id`,
which is already stored in the LangGraph checkpoint and the run registry, a resumed invocation
re-joins the trace without any extra persisted tracing state.

`execution_kind` says why the graph ran: `start` (a new request), `resume` (an answer to an
interrupt; `interrupt_kind` names it), `recovery` (a recovery action or legacy bridge) or
`continue` (an unfinished checkpoint). The execution's output records the final stage and, when
the invocation stopped at an interrupt, `paused_at`.

## Tree of one run (captured from the real pipeline with fake models)

Spans exported by the production code path in a run that is approved twice (routing and parsing
child runs omitted for brevity):

```
trace bb977cf7…  session thread-7f3a
├─ research.execution [agent]  execution_kind=start           output: paused_at=plan_approval
│  └─ research_graph [chain]
│     ├─ clarify_with_user [chain]
│     │  ├─ model.guardrail [chain] → ChatModel [generation]
│     │  └─ model.scope [chain] → ChatModel [generation]
│     ├─ write_research_brief → model.scope → [generation]
│     ├─ planning → model.planner → [generation]
│     └─ await_plan_approval [chain]  (level DEFAULT: an interrupt is a pause, not an error)
├─ research.execution [agent]  execution_kind=resume interrupt_kind=plan_approval
│  └─ research_graph
│     ├─ await_plan_approval
│     ├─ researching
│     │  ├─ research_agent [agent] todo_id=79aa…        ┐ sub-agents run concurrently
│     │  │  ├─ llm_call → model.researcher [generation]  │ (asyncio.to_thread); each one
│     │  │  ├─ tool_node → mcp.search [tool]             │ nests under researching, never
│     │  │  └─ compress_research → model.research_compressor → [generation]
│     │  ├─ research_agent [agent] todo_id=3eba…        ┘ as a separate root
│     │  ├─ research.persist [span] todo_id=3eba…   (VFS writes + vector_index.upsert)
│     │  └─ research.persist [span] todo_id=79aa…
│     ├─ research_complete
│     ├─ synthesizing → model.synthesizer → [generation]
│     ├─ verifying
│     │  ├─ model.verify → [generation]
│     │  └─ ragas.faithfulness [evaluator]   → trace score "faithfulness"
│     ├─ evaluating_quality
│     └─ approval           output: paused_at=publication_approval
└─ research.execution [agent]  execution_kind=resume interrupt_kind=publication_approval
   └─ research_graph
      ├─ approval
      └─ publishing
```

In the Langfuse UI this is one trace with three top-level executions, ordered by time, and one
session that lists it. Node observations carry `node`, `stage`, `draft_version`,
`improvement_cycle` and `plan_version`. Worker, tool and persistence observations carry
`todo_id`. Every observation carries the run and thread through trace-level metadata.

## How each part joins the tree

| Part | Mechanism |
|---|---|
| Graph and nodes | `RunTracer.execution()` creates a Langfuse `CallbackHandler` per invocation; `Execution.graph_config()` puts it in the LangGraph config. LangGraph emits one run per node. `node_wrapper` adds metadata to that run with `update_current_span`. It no longer opens a second span |
| LLM calls | `ModelRouter` passes only `run_name="model.<role>"` and metadata. The call inherits the callback of the node or sub-agent it runs in, so the handler records it once as a generation with model, usage and latency. Structured-output parsing is part of the chain, so an invalid answer is an ERROR on that attempt's `model.<role>` run |
| Research sub-agents | `ResearchAgent.research` invokes its graph with `run_name="research_agent"` and `todo_id`. The supervisor runs them with `asyncio.to_thread`, which copies `contextvars` (the OpenTelemetry context and LangChain's parent run config) into the worker thread |
| MCP, persistence, vector index, RAGAS | `traced(name, as_type=...)` opens a Langfuse observation as a child of the current one. The active client travels in a `ContextVar`, so this also works inside `asyncio.run`, `anyio.run`, `gather` and worker threads. Outside an execution `traced` does nothing |
| RAGAS judge (OpenAI) | Inside an execution the judge uses `langfuse.openai.AsyncOpenAI`, so each judge call is a generation under its evaluator observation |
| Recovery, continuation, legacy bridges | `WorkflowRecoveryService(graph, registry, tracer)` wraps every `graph.invoke` in an execution |
| Scores | `node_wrapper` sends `faithfulness` (from `verifying`) and `context_precision` / `response_relevancy` (from `evaluating_quality`) as trace scores, each only from the node that computes it |

## Data that leaves the process

The client is created with `mask=`. Every input, output and metadata value, including what the
LangChain callback records, is redacted before export: credential-named keys,
`key=value` secrets, `Bearer` tokens and secret URL query parameters such as
`?tavilyApiKey=` are replaced with `[REDACTED]`, and strings longer than 4,000 characters are
truncated. MCP observations export the query and the result URLs, not page content (which stays
in the VFS). Tracing never raises into the workflow. If Langfuse is disabled, has no keys, or
fails authentication, `RunTracer` is inert and `disabled_reason` says why. The app shows that
reason and runs unchanged. Spans are exported in the background; a chat turn never waits for
Langfuse.

## Limits of the installed SDK and integrations

- An observation cannot stay open across a human pause and still survive a restart, so the
  executions of a run are siblings in one trace, not children of one long run span.
- Spans created with plain OpenTelemetry tracers are dropped by Langfuse's default export filter.
  That is why non-LangChain operations use the Langfuse client (`traced`) and not an
  OpenTelemetry tracer.
- Gemini RAGAS judge calls go through `google-genai` and instructor, which have no Langfuse
  integration. Only their `ragas.<metric>` evaluator observation and its score are recorded.
- Embedding calls (`embed_documents` / `embed_query`) are plain LangChain embedding methods, not
  runnables. They appear as `vector_index.upsert` / `vector_index.query` observations without
  token usage or cost.
- Conditional-edge routers (`route_after_*`) appear as small chain runs under the node they
  follow. LangGraph emits them; they show which route was taken.
