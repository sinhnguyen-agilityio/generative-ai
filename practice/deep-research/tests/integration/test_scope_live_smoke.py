import os
import pytest
from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.config import load_local_env
from intelligence_collector.graph.build import build_graph
from intelligence_collector.models.factory import ModelConfig, ModelFactory
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.observability.tracing import Tracer


def test_scope_live_llm_smoke(tmp_path):
    load_local_env()
    if os.getenv("RUN_LIVE_LLM") != "1":
        pytest.skip("set RUN_LIVE_LLM=1 to enable the opt-in live LLM smoke test")
    model_name = os.getenv("INTELLIGENCE_REASONING_MODEL")
    if not model_name:
        pytest.skip("Set INTELLIGENCE_REASONING_MODEL when RUN_LIVE_LLM=1")
    provider = os.getenv("INTELLIGENCE_REASONING_PROVIDER") or None
    model = ModelFactory().create(ModelConfig(model_name, provider=provider, timeout=45))
    tracer = Tracer.from_env(required=True)
    agent = ScopeAgent(
        ModelRouter(
            model,
            trace=tracer.model_event,
            callbacks=tracer.callbacks,
        )
    )
    from intelligence_collector.storage.vfs import VFS
    graph = build_graph(scope_agent=agent, vfs=VFS(tmp_path), tracer=tracer)
    request = "Compare PostgreSQL and SQLite for local Python applications. Use technical primary sources, define success as evidence-backed factual claims, and return a technical report."
    with tracer.execution("live-smoke", "live-smoke", input={"request": request}) as execution:
        result = graph.invoke({
            "run_id": "live-smoke",
            "thread_id": "live-smoke",
            "request": request,
        })
        execution.output = {"stage": result["stage"]}
    assert result["scope_assessment"]["need_clarification"] is False
    assert result["stage"] == "Planning"
    assert execution.trace_id
