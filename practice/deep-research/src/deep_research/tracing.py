"""Langfuse tracing: one trace per research thread, scores attached to it. Off unless enabled."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _trace_id(thread_id: str) -> str:
    """Derive a stable Langfuse trace identifier from a research thread.

    Args:
        thread_id: Research thread identifier.

    Returns:
        Deterministic trace ID shared by all turns of the thread.
    """
    from langfuse import Langfuse

    return Langfuse.create_trace_id(seed=thread_id)


def langfuse_config(enabled: bool, thread_id: str) -> dict:
    """Build Langfuse callbacks and metadata for a shared thread trace.

    Args:
        enabled: Whether tracing is enabled.
        thread_id: Thread identifier used as trace seed and session ID.

    Returns:
        Callbacks and metadata, or an empty dictionary when disabled.
    """
    if not enabled:
        return {}
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    get_client()  # reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
    handler = CallbackHandler(trace_context={"trace_id": _trace_id(thread_id)})
    return {"callbacks": [handler], "metadata": {"langfuse_session_id": thread_id}}


def score(enabled: bool, thread_id: str, name: str, value: float, comment: str = "") -> None:
    """Record a named score on the thread trace without failing research on score errors.

    Args:
        enabled: Whether tracing is enabled.
        thread_id: Research thread receiving the score.
        name: Metric name.
        value: Numeric metric value.
        comment: Optional explanation attached to the score.
    """
    if not enabled:
        return
    try:
        from langfuse import get_client

        get_client().create_score(
            trace_id=_trace_id(thread_id), name=name, value=value, comment=comment or None
        )
    except Exception:  # tracing never breaks a run
        logger.exception("Langfuse: could not record score %s", name)


def flush(enabled: bool) -> None:
    """Flush queued Langfuse events when tracing is enabled.

    Args:
        enabled: Whether to flush the Langfuse client.
    """
    if enabled:
        from langfuse import get_client

        get_client().flush()
