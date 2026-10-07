"""Langfuse tracing: one trace per research thread, scores attached to it. Off unless enabled."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _trace_id(thread_id: str) -> str:
    from langfuse import Langfuse

    return Langfuse.create_trace_id(seed=thread_id)


def langfuse_config(enabled: bool, thread_id: str) -> dict:
    """Callbacks and metadata to merge into the graph config. Every turn of the thread (start,
    resume after review) lands in the same trace and session."""
    if not enabled:
        return {}
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    get_client()  # reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_BASE_URL
    handler = CallbackHandler(trace_context={"trace_id": _trace_id(thread_id)})
    return {"callbacks": [handler], "metadata": {"langfuse_session_id": thread_id}}


def score(enabled: bool, thread_id: str, name: str, value: float, comment: str = "") -> None:
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
    if enabled:
        from langfuse import get_client

        get_client().flush()
