"""Langfuse tracing for research runs (Langfuse Python SDK v4, OpenTelemetry based).

Trace boundary
--------------
One research run is one Langfuse trace. Its ID is derived from the run ID
(``create_trace_id(seed=...)``), so any process can attach later work to it: nothing has to
stay open in memory while the graph waits for a human. ``session_id`` is the LangGraph thread
ID, which groups the run with anything else recorded for the same chat thread.

Every graph invocation (a new request, a resume after an interrupt, a recovery action, a
continuation) opens one ``research.execution`` observation in that trace and closes it when
the invocation returns, including when the graph pauses at an interrupt. The Langfuse
LangChain ``CallbackHandler`` created for that execution is passed in the graph's config, so
LangGraph nodes, nested graphs (research sub-agents) and LangChain model calls become its
descendants. Code that is not a LangChain runnable (MCP calls, persistence, vector index,
RAGAS) opens its own observation with ``traced()``, which nests under whatever observation is
current, also inside asyncio tasks and worker threads, because both the OpenTelemetry context
and the active client travel in ``contextvars``.

    trace  (research run; session = thread)
      research.execution  [kind=start]
        research_graph                     LangGraph root run
          <node>                           one per graph node
            model.<role>                   LangChain chain -> generation
            research_agent  [todo_id]      sub-agent graph, one per worker task
              tool_node -> mcp.search      MCP tool call
            research.persist [todo_id]     VFS + vector index writes
      research.execution  [kind=resume, interrupt=plan_approval]
        research_graph -> ...

Tracing never decides workflow behaviour: when Langfuse is disabled or misconfigured the
tracer is inert and the graph runs exactly the same.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import logging
from typing import Any
from uuid import uuid4

from intelligence_collector.config import LangfuseSettings
from intelligence_collector.models.redaction import SECRET_MARKERS, redact_text

logger = logging.getLogger(__name__)

TRACE_NAME = "deep-research"
TRACE_TAGS = ["deep-research", "langgraph"]
PAUSED_MESSAGE = "Paused: waiting for a human decision"
# Longest string exported to Langfuse; prompts carry source excerpts, which stay in the VFS.
MAX_TRACE_TEXT = 4000

# The Langfuse client of the execution in progress. A ContextVar, so asyncio tasks and threads
# started with copied contexts (asyncio.to_thread, asyncio.run, anyio.run) see the same client.
_ACTIVE_CLIENT: ContextVar[Any] = ContextVar("langfuse_active_client", default=None)


# ===== PAYLOAD MASKING =====


def _truncate(text: str) -> str:
    if len(text) <= MAX_TRACE_TEXT:
        return text
    return f"{text[:MAX_TRACE_TEXT]}… [truncated {len(text) - MAX_TRACE_TEXT} chars]"


def mask_payload(value: Any) -> Any:
    """Redact credential-shaped keys and values and cut long strings. Messages become
    ``{"role", "content"}`` so state snapshots stay readable."""
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if any(marker in str(key).lower() for marker in SECRET_MARKERS)
            else mask_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [mask_payload(item) for item in value]
    if isinstance(value, str):
        return _truncate(redact_text(value))
    if hasattr(value, "type") and hasattr(value, "content") and not isinstance(value, type):
        return {"role": str(value.type), "content": mask_payload(value.content)}
    return value


def _langfuse_mask(*, data: Any, **_: Any) -> Any:
    """``Langfuse(mask=...)`` hook: applied to every input, output and metadata the SDK sends,
    including what the LangChain callback records."""
    return mask_payload(data)


# ===== OBSERVATIONS OUTSIDE LANGCHAIN =====


@dataclass
class Observation:
    """Handle yielded by ``traced()``: set ``output`` (and optionally ``metadata``) before the
    block ends. Inert when tracing is off."""

    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _is_pause(exc: BaseException) -> bool:
    """LangGraph's interrupt unwinds as an exception but is an intended pause, not a failure."""
    try:
        from langgraph.errors import GraphBubbleUp
    except ImportError:  # pragma: no cover - langgraph is a dependency
        return False
    return isinstance(exc, GraphBubbleUp)


@contextmanager
def traced(
    name: str, *, as_type: str = "span", input: Any = None, metadata: dict | None = None,
) -> Iterator[Observation]:
    """Record a non-LangChain operation (MCP call, persistence, retrieval, evaluation) as a child
    of the current observation. An exception marks this observation as ERROR and propagates;
    completed siblings are unaffected. No-op outside a traced execution."""
    handle = Observation(metadata=dict(metadata or {}))
    client = _ACTIVE_CLIENT.get()
    if client is None:
        yield handle
        return
    try:
        manager = client.start_as_current_observation(
            name=name, as_type=as_type, input=input, metadata=handle.metadata)
        observation = manager.__enter__()
    except Exception:  # tracing must never break the operation
        logger.exception("Langfuse: could not start observation %s", name)
        yield handle
        return
    error: BaseException | None = None
    try:
        yield handle
    except BaseException as exc:
        error = exc
        raise
    finally:
        try:
            updates: dict[str, Any] = {"metadata": handle.metadata}
            if handle.output is not None:
                updates["output"] = handle.output
            if error is not None and _is_pause(error):
                updates["status_message"] = PAUSED_MESSAGE
            elif error is not None:
                updates.update(level="ERROR",
                               status_message=f"{type(error).__name__}: {redact_text(error)}")
            observation.update(**updates)
        finally:
            manager.__exit__(None, None, None)


def tracing_active() -> bool:
    """True while a traced execution is in progress in this context."""
    return _ACTIVE_CLIENT.get() is not None


def annotate_current(**metadata: Any) -> None:
    """Add low-cardinality metadata to the current observation (e.g. the LangGraph node run the
    callback opened). Skips ``None`` values; no-op outside a traced execution."""
    client = _ACTIVE_CLIENT.get()
    values = {key: value for key, value in metadata.items() if value not in (None, "")}
    if client is None or not values or client.get_current_observation_id() is None:
        return
    try:
        client.update_current_span(metadata=values)
    except Exception:
        logger.exception("Langfuse: could not annotate the current observation")


def score_current_trace(name: str, value: float, *, comment: str = "") -> None:
    """Attach a numeric score (e.g. RAGAS faithfulness) to the run's trace."""
    client = _ACTIVE_CLIENT.get()
    if client is None:
        return
    try:
        client.score_current_trace(
            name=name, value=value, data_type="NUMERIC", comment=comment or None)
    except Exception:
        logger.exception("Langfuse: could not score the trace")


# ===== GRAPH EXECUTIONS =====


def _interrupt_kind(result: dict) -> str:
    for item in result.get("__interrupt__") or ():
        value = getattr(item, "value", None)
        if isinstance(value, dict) and value.get("kind"):
            return str(value["kind"])
    return ""


@dataclass
class Execution:
    """One graph invocation inside the run's trace. ``graph_config`` adds the execution's
    callback and correlation metadata to a LangGraph config; ``finish`` records the outcome."""

    run_id: str
    thread_id: str
    kind: str
    execution_id: str
    trace_id: str = ""
    callbacks: list[Any] = field(default_factory=list)
    outcome: dict[str, Any] = field(default_factory=dict)

    def graph_config(self, config: dict) -> dict:
        metadata = {**config.get("metadata", {}), "run_id": self.run_id,
                    "execution_id": self.execution_id, "execution_kind": self.kind}
        traced_config = {**config, "metadata": metadata, "run_name": "research_graph"}
        if self.callbacks:
            traced_config["callbacks"] = [*config.get("callbacks", []), *self.callbacks]
        return traced_config

    def finish(self, result: dict) -> None:
        """Summarise the final state: stage, report version and where the graph paused, if it
        did. (``status`` is left out: the state only updates it at some nodes.)"""
        self.outcome = {key: result.get(key) for key in (
            "stage", "draft_version", "verification_outcome", "feedback_route")
            if result.get(key) not in (None, "")}
        paused_at = _interrupt_kind(result)
        if paused_at:
            self.outcome["paused_at"] = paused_at


class RunTracer:
    """Owns the Langfuse client and opens one ``research.execution`` per graph invocation."""

    def __init__(self, client=None, *, public_key: str = "", disabled_reason: str = ""):
        self.client = client
        self.public_key = public_key  # selects this client for the LangChain callback
        self.disabled_reason = disabled_reason

    @property
    def enabled(self) -> bool:
        return self.client is not None

    @classmethod
    def from_settings(cls, settings: LangfuseSettings, *, span_exporter=None) -> "RunTracer":
        """A tracer for `settings`. Disabled settings give an inert tracer; a client that cannot
        be created or authenticated also gives an inert tracer whose ``disabled_reason`` says why,
        so a tracing problem never stops research. ``span_exporter`` replaces the network
        exporter (tests capture spans in memory and skip the auth check)."""
        if not settings.enabled:
            return cls()
        try:
            from langfuse import Langfuse

            client = Langfuse(
                public_key=settings.public_key,
                secret_key=settings.secret_key,
                base_url=settings.host,
                environment=settings.environment,
                release=settings.release,
                mask=_langfuse_mask,
                span_exporter=span_exporter,
            )
            if span_exporter is None and not client.auth_check():
                raise RuntimeError(
                    "Langfuse authentication failed. Verify that LANGFUSE_PUBLIC_KEY and "
                    "LANGFUSE_SECRET_KEY belong to the same project and that LANGFUSE_BASE_URL "
                    "matches that project's region.")
        except Exception as exc:
            reason = redact_text(exc)
            logger.warning("Langfuse tracing disabled: %s", reason)
            return cls(disabled_reason=reason)
        return cls(client, public_key=settings.public_key)

    @classmethod
    def from_env(cls) -> "RunTracer":
        try:
            settings = LangfuseSettings.from_env()
        except RuntimeError as exc:  # enabled without credentials
            logger.warning("Langfuse tracing disabled: %s", exc)
            return cls(disabled_reason=str(exc))
        return cls.from_settings(settings)

    def trace_id_for(self, run_id: str) -> str:
        """The run's trace ID: the same in every process, so resumed work rejoins the trace."""
        if self.client is None:
            return ""
        return self.client.create_trace_id(seed=f"research-run:{run_id}")

    @contextmanager
    def execution(
        self, *, run_id: str, thread_id: str, kind: str, input: Any = None,
        interrupt_kind: str = "",
    ) -> Iterator[Execution]:
        """Open ``research.execution`` for one graph invocation and make it current.

        `kind` is ``start``, ``resume``, ``recovery`` or ``continue``; `interrupt_kind` names the
        interrupt a resume answers (``plan_approval``, ``publication_approval``, ...)."""
        execution = Execution(run_id=run_id, thread_id=thread_id, kind=kind,
                              execution_id=str(uuid4()), trace_id=self.trace_id_for(run_id))
        if self.client is None:
            yield execution
            return
        from langfuse import propagate_attributes
        from langfuse.langchain import CallbackHandler

        metadata = {"run_id": run_id, "thread_id": thread_id, "execution_id": execution.execution_id,
                    "execution_kind": kind, "interrupt_kind": interrupt_kind}
        # A fresh handler per execution: no callback state is shared between invocations,
        # Streamlit reruns or concurrently open runs.
        execution.callbacks = [CallbackHandler(public_key=self.public_key or None)]
        token = _ACTIVE_CLIENT.set(self.client)
        try:
            with self.client.start_as_current_observation(
                name="research.execution",
                as_type="agent",
                trace_context={"trace_id": execution.trace_id},
                input=input,
                metadata={key: value for key, value in metadata.items() if value},
            ) as observation, propagate_attributes(
                session_id=thread_id,
                trace_name=TRACE_NAME,
                tags=TRACE_TAGS,
                metadata={"run_id": run_id, "thread_id": thread_id},
            ):
                try:
                    yield execution
                except BaseException as exc:
                    if not _is_pause(exc):
                        observation.update(level="ERROR", status_message=(
                            f"{type(exc).__name__}: {redact_text(exc)}"))
                    raise
                finally:
                    observation.update(output=execution.outcome or None)
                    if execution.outcome.get("paused_at"):
                        observation.update(status_message=(
                            f"{PAUSED_MESSAGE} ({execution.outcome['paused_at']})"))
        finally:
            # No synchronous flush: the SDK exports in the background (and at exit), so an
            # unreachable Langfuse never delays a chat turn.
            _ACTIVE_CLIENT.reset(token)

    def score(self, name: str, value: float, *, comment: str = "") -> None:
        score_current_trace(name, value, comment=comment)

    def flush(self) -> None:
        if self.client is not None:
            try:
                self.client.flush()
            except Exception:
                logger.exception("Langfuse: flush failed")
