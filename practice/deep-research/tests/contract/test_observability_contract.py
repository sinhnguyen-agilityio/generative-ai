from contextlib import contextmanager

import pytest

from intelligence_collector.config import LangfuseSettings
from intelligence_collector.observability.tracing import Tracer


class FakeObservation:
    def __init__(self, record, client=None):
        self.record = record
        self.client = client
        self.id = record["id"]
        self.trace_id = record.get("trace_context", {}).get("trace_id", "")

    def update(self, **values):
        self.record.setdefault("updates", []).append(values)

    def end(self):
        self.record["ended"] = True

    def start_as_current_observation(self, **values):
        """A child observation of this one (as Langfuse's ``LangfuseSpan`` does)."""
        return self.client.start_as_current_observation(_parent=self.record, **values)


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

    def _record(self, values, parent):
        record = {**values, "id": f"span-{len(self.records)}", "parent": parent and parent["name"]}
        self.records.append(record)
        return record

    def start_observation(self, **values):
        """An observation that is not made current and stays open until ``end()``."""
        return FakeObservation(self._record(values, None), self)

    @contextmanager
    def start_as_current_observation(self, _parent=None, **values):
        record = self._record(values, _parent or (self.stack[-1] if self.stack else None))
        self.stack.append(record)
        try:
            yield FakeObservation(record, self)
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
    assert client.records[0]["name"] == "research.turn"
    assert client.records[1]["name"] == "graph.scope"
    assert client.records[1]["parent"] == "research.turn"
    assert client.flush_count == 1


def test_executions_sharing_a_trace_key_land_in_one_trace(monkeypatch):
    """One chat thread = one trace: each turn shares the thread's trace ID."""
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
    roots = [r for r in client.records if r["name"] == "research.turn"]
    assert [r["trace_context"]["trace_id"] for r in roots] == [
        "trace:thread:t1", "trace:thread:t1", "trace:thread:t2",
    ]


def test_turns_of_a_thread_nest_under_one_open_thread_root(monkeypatch):
    """The thread root is opened once, stays open across turns (so Langfuse times the whole
    thread), parents every turn directly, and ends only on ``end_thread``."""
    monkeypatch.setattr("langfuse.propagate_attributes", fake_propagation)
    client = FakeLangfuseClient()
    tracer = Tracer(client=client)

    for turn in ("turn-1", "turn-2"):
        tracer.start_thread("run", "thread", trace_key="thread:t1", input={"request": "q"})
        with tracer.execution("run", "thread", execution_id=turn, trace_key="thread:t1") as ex:
            ex.output = {"stage": turn}
        assert ex.trace_id == "trace:thread:t1"

    root, *turns = client.records
    assert root["name"] == "research.thread"
    assert root["trace_context"] == {"trace_id": "trace:thread:t1"}
    assert "ended" not in root
    assert [(t["name"], t["parent"]) for t in turns] == [
        ("research.turn", "research.thread"), ("research.turn", "research.thread"),
    ]
    # A child turn must not carry trace_context: Langfuse would mark it as a trace root.
    assert all("trace_context" not in t for t in turns)
    assert '"turn-2"' in root["updates"][-1]["output"]

    tracer.end_thread("thread")
    tracer.end_thread("thread")  # idempotent
    assert root["ended"] is True


def test_open_thread_roots_are_ended_when_the_tracer_is_collected(monkeypatch):
    import gc

    monkeypatch.setattr("langfuse.propagate_attributes", fake_propagation)
    client = FakeLangfuseClient()
    tracer = Tracer(client=client)
    tracer.start_thread("run", "thread", trace_key="thread:t1")

    del tracer
    gc.collect()

    assert client.records[0]["ended"] is True


def test_thread_root_is_local_only_when_tracing_is_off():
    tracer = Tracer()

    tracer.start_thread("run", "thread", trace_key="thread:t1")
    tracer.start_thread("run", "thread", trace_key="thread:t1")
    tracer.end_thread("thread")

    assert [s.name for s in tracer.spans] == ["research.thread"]


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
        "evaluating_quality",
        lambda state: {**state, "faithfulness_score": 0.9, "faithfulness_error": ""},
        tracer,
    )

    node({"run_id": "run", "thread_id": "thread"})

    assert client.scores[0]["name"] == "faithfulness" and client.scores[0]["value"] == 0.9
    output = json.loads(tracer.spans[-1].output)
    assert output["faithfulness_score"] == 0.9
