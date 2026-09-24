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


def test_trace_payload_redacts_credential_shaped_fields(monkeypatch):
    monkeypatch.setattr("langfuse.propagate_attributes", fake_propagation)
    client = FakeLangfuseClient()
    tracer = Tracer(client=client)

    with tracer.execution("run", "thread", input={"api_key": "do-not-export"}):
        pass

    assert client.records[0]["input"] == {"api_key": "[REDACTED]"}
    assert "do-not-export" not in repr(client.records)


def test_required_live_settings_fail_before_client_creation(monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_LANGFUSE_ENABLED", "true")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="credentials are not configured"):
        LangfuseSettings.from_env(required=True)
