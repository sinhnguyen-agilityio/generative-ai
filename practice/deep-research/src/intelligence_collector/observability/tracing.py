from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
from typing import Any, Iterator
from uuid import uuid4

from intelligence_collector.config import LangfuseSettings


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

    @classmethod
    def from_settings(cls, settings: LangfuseSettings):
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
        return cls.from_settings(LangfuseSettings.from_env(required=required))

    @property
    def callbacks(self) -> list[Any]:
        return [self.callback_handler] if self.callback_handler is not None else []

    @contextmanager
    def execution(
        self,
        run_id: str,
        thread_id: str,
        *,
        input: Any = None,
        execution_id: str | None = None,
    ) -> Iterator[TraceSpan]:
        execution_id = execution_id or str(uuid4())
        attributes = {"run_id": run_id, "thread_id": thread_id, "execution_id": execution_id}
        local = TraceSpan(
            "research.execution",
            run_id,
            thread_id,
            attributes,
            execution_id=execution_id,
        )
        self.spans.append(local)
        if self.client is None:
            yield local
            return

        from langfuse import propagate_attributes

        trace_id = self.client.create_trace_id(seed=f"{run_id}:{execution_id}")
        local.trace_id = trace_id
        try:
            with self.client.start_as_current_observation(
                as_type="agent",
                name=local.name,
                trace_context={"trace_id": trace_id},
                input=_text_payload(input),
                metadata=_safe(attributes),
            ) as remote:
                with propagate_attributes(
                    session_id=thread_id,
                    trace_name="intelligence-collector",
                    metadata=_safe(attributes),
                    tags=["deep-research", "langgraph"],
                ):
                    try:
                        yield local
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
            except Exception as exc:
                local.error = f"{type(exc).__name__}: {exc}"
                remote.update(level="ERROR", status_message=local.error)
                raise
            finally:
                remote.update(output=_text_payload(local.output), metadata=_safe(local.attributes))

    def model_event(self, **event: Any) -> None:
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
        if self.client is not None:
            self.client.flush()
