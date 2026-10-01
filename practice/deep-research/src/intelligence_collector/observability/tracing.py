"""Langfuse tracing: one trace per chat thread with a single ``research.thread`` root, one
``research.turn`` span per chat turn beneath it, nested node spans and scores, plus a local,
credential-free record of every span for tests.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from typing import Any, Iterator
from uuid import uuid4

from langgraph.errors import GraphInterrupt

from intelligence_collector.config import LangfuseSettings


PAUSED_MESSAGE = "Paused: waiting for a human decision"
SECRET_KEYS = ("api_key", "secret", "authorization", "credential", "password", "token")


def _safe(value: Any) -> Any:
    """Recursively remove credential-shaped fields before telemetry export."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(marker in key.lower() for marker in SECRET_KEYS) else _safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _text_payload(value: Any) -> str:
    """Keep Langfuse observation inputs/outputs human-readable and searchable."""
    safe = _safe(value)
    if isinstance(safe, str):
        return safe
    return json.dumps(safe, ensure_ascii=False, indent=2, default=str)


@dataclass
class TraceSpan:
    """Local record of one span: what was sent to Langfuse, with credentials redacted."""

    name: str
    run_id: str
    thread_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: str = ""
    trace_id: str = ""
    execution_id: str = ""


class Tracer:
    """Langfuse v4 tracing boundary with an inspectable credential-free local record."""

    def __init__(self, client=None, callback_handler=None):
        self.client = client
        self.callback_handler = callback_handler
        self.spans: list[TraceSpan] = []
        self.scores: list[dict[str, Any]] = []

    @classmethod
    def from_settings(cls, settings: LangfuseSettings):
        """Create a tracer from settings; disabled settings give a local-only tracer. Raises
        ``RuntimeError`` if Langfuse rejects the keys.
        """
        if not settings.enabled:
            return cls()
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler

        client = Langfuse(
            public_key=settings.public_key,
            secret_key=settings.secret_key,
            base_url=settings.host,
            environment=settings.environment,
            release=settings.release,
        )
        if not client.auth_check():
            raise RuntimeError(
                "Langfuse authentication failed. Verify that LANGFUSE_PUBLIC_KEY and "
                "LANGFUSE_SECRET_KEY belong to the same project and that LANGFUSE_BASE_URL "
                "matches that project's region."
            )
        return cls(client=client, callback_handler=CallbackHandler(public_key=settings.public_key))

    @classmethod
    def from_env(cls, *, required: bool = False):
        """``from_settings`` with the settings read from the environment."""
        return cls.from_settings(LangfuseSettings.from_env(required=required))

    @property
    def callbacks(self) -> list[Any]:
        """LangChain callbacks that report model calls to Langfuse (empty when tracing is off)."""
        return [self.callback_handler] if self.callback_handler is not None else []

    def thread_root(self, run_id: str, thread_id: str, *, trace_key: str, input: Any = None) -> str:
        """Create the single ``research.thread`` parent observation of a chat thread's trace and
        return its span ID (empty when tracing is off). It ends immediately: it only groups the
        thread's turns, which attach to it later through ``execution(parent_span_id=...)``, so a
        restart can never leave an unexported parent behind."""
        local = TraceSpan("research.thread", run_id, thread_id, {"run_id": run_id, "thread_id": thread_id})
        self.spans.append(local)
        if self.client is None:
            return ""
        local.trace_id = self.client.create_trace_id(seed=trace_key)
        try:
            with self._trace_attributes(thread_id, local.attributes):
                with self.client.start_as_current_observation(
                    as_type="agent",
                    name=local.name,
                    trace_context={"trace_id": local.trace_id},
                    input=_text_payload(input),
                    metadata=_safe(local.attributes),
                ) as remote:
                    return remote.id
        finally:
            self.flush()

    @staticmethod
    def _trace_attributes(thread_id: str, attributes: dict[str, Any]):
        """Trace-level attributes shared by every observation of a thread's trace."""
        from langfuse import propagate_attributes

        return propagate_attributes(
            session_id=thread_id,
            trace_name="intelligence-collector",
            metadata=_safe(attributes),
            tags=["deep-research", "langgraph"],
        )

    @contextmanager
    def execution(
        self,
        run_id: str,
        thread_id: str,
        *,
        input: Any = None,
        execution_id: str | None = None,
        trace_key: str | None = None,
        parent_span_id: str | None = None,
    ) -> Iterator[TraceSpan]:
        """One ``research.turn`` span per execution (e.g. one chat turn). Executions that share a
        `trace_key` land in the same Langfuse trace; with a `parent_span_id` (from
        ``thread_root``) they nest under that one parent instead of being sibling roots. Without a
        `trace_key`, every execution gets its own trace."""
        execution_id = execution_id or str(uuid4())
        attributes = {"run_id": run_id, "thread_id": thread_id, "execution_id": execution_id}
        local = TraceSpan(
            "research.turn",
            run_id,
            thread_id,
            attributes,
            execution_id=execution_id,
        )
        self.spans.append(local)
        if self.client is None:
            yield local
            return

        trace_id =self.client.create_trace_id(seed=trace_key or f"{run_id}:{execution_id}")
        local.trace_id = trace_id
        trace_context = {"trace_id": trace_id}
        if parent_span_id:
            trace_context["parent_span_id"] = parent_span_id
        try:
            with self.client.start_as_current_observation(
                as_type="agent",
                name=local.name,
                trace_context=trace_context,
                input=_text_payload(input),
                metadata=_safe(attributes),
            ) as remote:
                with self._trace_attributes(thread_id, attributes):
                    try:
                        yield local
                    except GraphInterrupt:
                        # Not a failure: the graph paused for a human decision (the approval gate).
                        local.attributes["paused"] = True
                        remote.update(status_message=PAUSED_MESSAGE)
                        raise
                    except Exception as exc:
                        local.error = f"{type(exc).__name__}: {exc}"
                        remote.update(level="ERROR", status_message=local.error)
                        raise
                    finally:
                        remote.update(output=_text_payload(local.output), metadata=_safe(local.attributes))
        finally:
            self.flush()

    @contextmanager
    def span(
        self,
        name: str,
        run_id: str,
        thread_id: str,
        *,
        as_type: str = "span",
        input: Any = None,
        **attributes: Any,
    ) -> Iterator[TraceSpan]:
        """Open a child span under the current execution. Yields the local record; its ``output`` is
        sent when the block ends, and an exception marks the span as an error (a human-review pause
        does not).
        """
        local = TraceSpan(name, run_id, thread_id, _safe(attributes))
        self.spans.append(local)
        if self.client is None:
            yield local
            return
        with self.client.start_as_current_observation(
            as_type=as_type,
            name=name,
            input=_text_payload(input),
            metadata={"run_id": run_id, "thread_id": thread_id, **_safe(attributes)},
        ) as remote:
            try:
                yield local
            except GraphInterrupt:
                # `interrupt()` unwinds through the node as an exception, but it is an intentional
                # pause for a human decision, so it must not show up as an ERROR in Langfuse.
                local.attributes["paused"] = True
                remote.update(status_message=PAUSED_MESSAGE)
                raise
            except Exception as exc:
                local.error = f"{type(exc).__name__}: {exc}"
                remote.update(level="ERROR", status_message=local.error)
                raise
            finally:
                remote.update(output=_text_payload(local.output), metadata=_safe(local.attributes))

    def score(self, name: str, value: float, *, comment: str = "") -> None:
        """Attach a numeric score (e.g. RAGAS faithfulness) to the current Langfuse trace, so it
        shows up in the trace's Scores and can be filtered/charted. Must be called while the
        trace's observation is active. Observability only: a failed export never raises."""
        record: dict[str, Any] = {"name": name, "value": value, "comment": comment}
        self.scores.append(record)
        if self.client is None:
            return
        try:
            self.client.score_current_trace(
                name=name, value=value, data_type="NUMERIC", comment=comment or None,
            )
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"

    def model_event(self, **event: Any) -> None:
        """Record a model validation event (e.g. a structured-output retry) as its own span."""
        run_id = str(event.get("run_id", ""))
        thread_id = str(event.get("thread_id", ""))
        name = f"model.validation.{event.get('role', 'unknown')}"
        attributes = _safe(event)
        attributes.pop("run_id", None)
        attributes.pop("thread_id", None)
        input_text = attributes.pop("input_text", "")
        output_text = attributes.pop("output_text", "")
        attributes.pop("decision", None)
        with self.span(
            name,
            run_id,
            thread_id,
            as_type="generation",
            input=input_text,
            **attributes,
        ) as span:
            span.output = output_text

    def flush(self) -> None:
        """Send buffered events to Langfuse."""
        if self.client is not None:
            self.client.flush()
