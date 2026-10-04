"""A real Langfuse SDK client whose spans are captured in memory instead of being sent.

Everything else is the production path: the ``RunTracer``, the Langfuse ``CallbackHandler``,
the mask function and OpenTelemetry context propagation. Tests assert on the exported span
tree (trace IDs and parent span IDs), which is exactly what the Langfuse UI renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any
from uuid import uuid4

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from intelligence_collector.config import LangfuseSettings
from intelligence_collector.observability.tracing import RunTracer

_METADATA = "langfuse.observation.metadata."


def _decode(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return value


@dataclass
class CapturedSpan:
    name: str
    span_id: int
    parent_id: int | None
    trace_id: str
    type: str
    level: str
    status_message: str
    session_id: str
    metadata: dict[str, Any] = field(default_factory=dict)
    input: Any = None
    output: Any = None
    start: int = 0


class LangfuseCapture:
    """Owns a traced ``RunTracer`` and reads back the spans it exported."""

    def __init__(self):
        self.exporter = InMemorySpanExporter()
        # A unique public key gives this test its own Langfuse resource manager and exporter.
        settings = LangfuseSettings(enabled=True, public_key=f"pk-test-{uuid4()}",
                                    secret_key="sk-test", host="http://127.0.0.1:9")
        self.tracer = RunTracer.from_settings(settings, span_exporter=self.exporter)
        assert self.tracer.enabled, self.tracer.disabled_reason
        # Scores are sent through Langfuse's REST API, not as spans: record them here instead,
        # with the trace they were attached to.
        self.scores: list[dict] = []
        client = self.tracer.client

        def score_current_trace(**score):
            self.scores.append({**score, "trace_id": client.get_current_trace_id()})

        client.score_current_trace = score_current_trace

    def spans(self) -> list[CapturedSpan]:
        self.tracer.flush()
        captured = []
        for span in self.exporter.get_finished_spans():
            attributes = dict(span.attributes or {})
            captured.append(CapturedSpan(
                name=span.name,
                span_id=span.context.span_id,
                parent_id=span.parent.span_id if span.parent else None,
                trace_id=f"{span.context.trace_id:032x}",
                type=str(attributes.get("langfuse.observation.type", "")),
                level=str(attributes.get("langfuse.observation.level", "")),
                status_message=str(attributes.get("langfuse.observation.status_message", "")),
                session_id=str(attributes.get("session.id", "")),
                metadata={key[len(_METADATA):]: _decode(value)
                          for key, value in attributes.items() if key.startswith(_METADATA)},
                input=_decode(attributes.get("langfuse.observation.input")),
                output=_decode(attributes.get("langfuse.observation.output")),
                start=span.start_time,
            ))
        return sorted(captured, key=lambda item: item.start)

    def named(self, name: str) -> list[CapturedSpan]:
        return [span for span in self.spans() if span.name == name]

    def parent(self, span: CapturedSpan) -> CapturedSpan | None:
        return next((item for item in self.spans() if item.span_id == span.parent_id), None)

    def ancestors(self, span: CapturedSpan) -> list[str]:
        """Names from the span's parent up to its execution root."""
        names, current = [], self.parent(span)
        while current is not None:
            names.append(current.name)
            current = self.parent(current)
        return names

    def render(self) -> str:
        """The span tree as indented text (handy in assertion messages and docs)."""
        spans = self.spans()
        ids = {span.span_id for span in spans}
        children: dict[int | None, list[CapturedSpan]] = {}
        for span in spans:
            children.setdefault(span.parent_id if span.parent_id in ids else None, []).append(span)
        lines: list[str] = []

        def walk(parent, depth):
            for span in children.get(parent, []):
                extra = {key: span.metadata[key] for key in ("todo_id", "execution_kind",
                         "interrupt_kind", "node") if key in span.metadata}
                level = f" level={span.level}" if span.level not in ("", "DEFAULT") else ""
                lines.append(f"{'  ' * depth}{span.name} [{span.type}]{level}"
                             + (f" {extra}" if extra else ""))
                walk(span.span_id, depth + 1)

        walk(None, 0)
        return "\n".join(lines)
