from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class TraceSpan:
    name: str
    run_id: str
    thread_id: str
    attributes: dict[str, Any] = field(default_factory=dict)


class Tracer:
    """Langfuse-compatible tracing boundary with a safe local no-op fallback."""

    def __init__(self, client=None):
        self.client = client
        self.spans: list[TraceSpan] = []

    @contextmanager
    def span(self, name: str, run_id: str, thread_id: str, **attributes: Any) -> Iterator[TraceSpan]:
        span = TraceSpan(name, run_id, thread_id, attributes)
        self.spans.append(span)
        if self.client is None:
            yield span
            return
        try:
            with self.client.start_as_current_observation(as_type="span", name=name) as remote:
                remote.update(metadata={"run_id": run_id, "thread_id": thread_id, **attributes})
                yield span
        except Exception:
            yield span
