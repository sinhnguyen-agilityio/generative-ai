from contextlib import contextmanager

import pytest

from intelligence_collector.config import LangfuseSettings
from intelligence_collector.observability.tracing import Tracer


class FakeObservation:
    def __init__(self, record):
        self.record = record

    def update(self, **values):
        self.record.setdefault("updates", []).append(values)


class FakeLangfuseClient:
    def __init__(self):
        self.records = []
        self.stack = []
        self.flush_count = 0
        self.scores = []

    def score_current_trace(self, **values):
        self.scores.append(values)

    @staticmethod
    def create_trace_id(*, seed):
        return f"trace:{seed}"

    @contextmanager
    def start_as_current_observation(self, **values):
        record = {**values, "parent": self.stack[-1]["name"] if self.stack else None}
        self.records.append(record)
        self.stack.append(record)
        try:
            yield FakeObservation(record)
        finally:
            self.stack.pop()

    def flush(self):
        self.flush_count += 1


@contextmanager
def fake_propagation(**kwargs):
    yield kwargs


def test_tracer_creates_root_thread_and_nested_spans(monkeypatch):
    monkeypatch.setattr("langfuse.propagate_attributes", fake_propagation)
    client = FakeLangfuseClient()
    tracer = Tracer(client=client)

    with tracer.execution("run", "thread", execution_id="turn-1") as root:
        with tracer.span("graph.scope", "run", "thread", node="scope") as node:
            node.output = {"stage": "Scoping"}
        root.output = {"stage": "Scoping"}

    assert root.trace_id == "trace:run:turn-1"
    assert client.records[0]["name"] == "research.execution"
    assert client.records[1]["name"] == "graph.scope"
    assert client.records[1]["parent"] == "research.execution"
    assert client.flush_count == 1


def test_executions_sharing_a_trace_key_land_in_one_trace(monkeypatch):
    """One chat thread = one trace: each turn is its own root span inside the same trace."""
    monkeypatch.setattr("langfuse.propagate_attributes", fake_propagation)
    client = FakeLangfuseClient()
    tracer = Tracer(client=client)

    with tracer.execution("run", "thread", execution_id="turn-1", trace_key="thread:t1") as first:
        pass
    with tracer.execution("run", "thread", execution_id="turn-2", trace_key="thread:t1") as second:
        pass
    with tracer.execution("run", "thread", execution_id="turn-1", trace_key="thread:t2") as other:
        pass

    assert first.trace_id == second.trace_id == "trace:thread:t1"
    assert other.trace_id == "trace:thread:t2"
    roots = [r for r in client.records if r["name"] == "research.execution"]
    assert [r["trace_context"]["trace_id"] for r in roots] == [
        "trace:thread:t1", "trace:thread:t1", "trace:thread:t2",
    ]


def test_trace_payload_redacts_credential_shaped_fields(monkeypatch):
    monkeypatch.setattr("langfuse.propagate_attributes", fake_propagation)
    client = FakeLangfuseClient()
    tracer = Tracer(client=client)

    with tracer.execution("run", "thread", input={"api_key": "do-not-export"}):
        pass

    assert client.records[0]["input"] == '{\n  "api_key": "[REDACTED]"\n}'
    assert "do-not-export" not in repr(client.records)


def test_required_live_settings_fail_before_client_creation(monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_LANGFUSE_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="credentials are not configured"):
        LangfuseSettings.from_env(required=True)


def test_langfuse_authentication_failure_is_actionable(monkeypatch):
    class UnauthenticatedClient:
        def __init__(self, **kwargs):
            pass

        def auth_check(self):
            return False

    monkeypatch.setattr("langfuse.Langfuse", UnauthenticatedClient)
    settings = LangfuseSettings(
        enabled=True,
        public_key="public-key",
        secret_key="secret-key",
    )

    with pytest.raises(RuntimeError, match="LANGFUSE_BASE_URL"):
        Tracer.from_settings(settings)


def test_graph_interrupt_is_a_pause_not_an_error(monkeypatch):
    from langgraph.errors import GraphInterrupt

    monkeypatch.setattr("langfuse.propagate_attributes", fake_propagation)
    client = FakeLangfuseClient()
    tracer = Tracer(client=client)

    with pytest.raises(GraphInterrupt):
        with tracer.execution("run", "thread"):
            with tracer.span("graph.approval", "run", "thread"):
                raise GraphInterrupt()

    updates = [u for record in client.records for u in record["updates"]]
    assert not any(u.get("level") == "ERROR" for u in updates)
    assert any("Paused" in str(u.get("status_message", "")) for u in updates)
    assert all(span.error == "" for span in tracer.spans)


def test_tracer_score_is_sent_to_the_current_langfuse_trace():
    client = FakeLangfuseClient()
    tracer = Tracer(client=client)

    tracer.score("faithfulness", 0.87, comment="ragas")

    assert client.scores == [
        {"name": "faithfulness", "value": 0.87, "data_type": "NUMERIC", "comment": "ragas"}
    ]
    assert tracer.scores[0]["value"] == 0.87


def test_node_wrapper_exports_faithfulness_as_score_and_span_output(monkeypatch):
    import json

    from intelligence_collector.graph.instrumentation import node_wrapper

    monkeypatch.setattr("langfuse.propagate_attributes", fake_propagation)
    client = FakeLangfuseClient()
    tracer = Tracer(client=client)
    node = node_wrapper(
        "evaluating_faithfulness",
        lambda state: {**state, "faithfulness_score": 0.9, "faithfulness_error": ""},
        tracer,
    )

    node({"run_id": "run", "thread_id": "thread"})

    assert client.scores[0]["name"] == "faithfulness" and client.scores[0]["value"] == 0.9
    output = json.loads(tracer.spans[-1].output)
    assert output["faithfulness_score"] == 0.9
