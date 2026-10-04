"""Contract of the tracing boundary (``observability/tracing.py``) on its own; the span tree of
real graph runs is covered by ``tests/integration/test_trace_topology.py``."""

import pytest
from langchain_core.messages import HumanMessage

from intelligence_collector.config import LangfuseSettings
from intelligence_collector.graph.instrumentation import node_wrapper
from intelligence_collector.observability.tracing import (
    MAX_TRACE_TEXT,
    Execution,
    RunTracer,
    mask_payload,
)
from tests.fixtures.langfuse_capture import LangfuseCapture


def test_required_live_settings_fail_before_client_creation(monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_LANGFUSE_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="credentials are not configured"):
        LangfuseSettings.from_env(required=True)


def test_disabled_settings_give_an_inert_tracer():
    tracer = RunTracer.from_settings(LangfuseSettings(enabled=False))

    assert not tracer.enabled and tracer.disabled_reason == ""
    with tracer.execution(run_id="run", thread_id="thread", kind="start") as execution:
        config = execution.graph_config({"configurable": {"thread_id": "thread"}})
    assert "callbacks" not in config
    assert execution.trace_id == ""


def test_authentication_failure_disables_tracing_with_an_actionable_reason(monkeypatch):
    monkeypatch.setattr("langfuse.Langfuse.auth_check", lambda self: False)

    tracer = RunTracer.from_settings(
        LangfuseSettings(
            enabled=True, public_key="pk-auth-test", secret_key="sk", host="http://127.0.0.1:9"
        )
    )

    assert not tracer.enabled
    assert "LANGFUSE_BASE_URL" in tracer.disabled_reason


def test_trace_id_is_derived_from_the_run_and_stable_across_clients():
    first, second = LangfuseCapture().tracer, LangfuseCapture().tracer

    assert first.trace_id_for("run-1") == second.trace_id_for("run-1")
    assert first.trace_id_for("run-1") != first.trace_id_for("run-2")


def test_graph_config_adds_the_execution_callback_and_correlation_metadata():
    execution = Execution(
        run_id="run",
        thread_id="thread",
        kind="resume",
        execution_id="exec-1",
        callbacks=["handler"],
    )

    config = execution.graph_config(
        {"configurable": {"thread_id": "thread"}, "metadata": {"source": "chat"}}
    )

    assert config["callbacks"] == ["handler"]
    assert config["run_name"] == "research_graph"
    assert config["metadata"] == {
        "source": "chat",
        "run_id": "run",
        "execution_id": "exec-1",
        "execution_kind": "resume",
    }
    assert config["configurable"] == {"thread_id": "thread"}


def test_finish_records_where_the_invocation_paused():
    class Interrupt:
        value = {"kind": "plan_approval", "plan": {"tasks": []}}

    execution = Execution(run_id="run", thread_id="thread", kind="start", execution_id="e")
    execution.finish(
        {
            "stage": "Planning",
            "status": "paused",
            "draft": "long text",
            "__interrupt__": (Interrupt(),),
        }
    )

    assert execution.outcome == {"stage": "Planning", "paused_at": "plan_approval"}


def test_mask_redacts_credentials_and_bounds_strings():
    masked = mask_payload(
        {
            "api_key": "sk-live",
            "nested": [{"Authorization": "Bearer abcdefghijkl"}],
            "url": "https://mcp.test/?tavilyApiKey=tvly-1&x=1",
            "messages": [HumanMessage(content="token=abc " + "y" * (MAX_TRACE_TEXT + 50))],
        }
    )

    assert masked["api_key"] == "[REDACTED]"
    assert masked["nested"] == [{"Authorization": "[REDACTED]"}]
    assert masked["url"] == "https://mcp.test/?tavilyApiKey=[REDACTED]&x=1"
    message = masked["messages"][0]
    assert message["role"] == "human"
    assert message["content"].startswith("token=[REDACTED] ")
    assert message["content"].endswith("[truncated 67 chars]")


def test_node_wrapper_annotates_the_node_and_scores_only_where_the_score_is_produced():
    capture = LangfuseCapture()
    verifying = node_wrapper(
        "verifying",
        lambda state: {**state, "faithfulness_score": 0.91, "draft_version": 2},
        capture.tracer,
    )
    approval = node_wrapper("approval", lambda state: state, capture.tracer)

    with capture.tracer.execution(run_id="run", thread_id="thread", kind="start"):
        from intelligence_collector.observability.tracing import traced

        with traced("verifying"):  # stands in for the node run the LangChain callback opens
            state = verifying({"run_id": "run", "thread_id": "thread"})
        with traced("approval"):
            approval(state)

    node = capture.named("verifying")[0]
    assert node.metadata["node"] == "verifying" and node.metadata["draft_version"] == 2
    assert [(s["name"], s["value"]) for s in capture.scores] == [("faithfulness", 0.91)]
