"""Expensive dependencies, built once per browser session (see ``runtime.py`` for the wiring).

They live in ``st.session_state`` rather than ``st.cache_resource``: the graph holds a SQLite
checkpoint connection and lifecycle callbacks bound to one session, so it must not be shared
across browser sessions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from intelligence_collector.config import Settings
from intelligence_collector.graph.recovery import WorkflowRecoveryService
from intelligence_collector.observability.tracing import RunTracer
from intelligence_collector.runtime import build_model_routers, build_runtime, build_scope_agent
from intelligence_collector.storage.vfs import VFS

LIFECYCLE_CALLBACK = "_lifecycle_callback"  # set by a running chat turn (see turn.py)


def _session_cached(key: str, build: Callable[[], Any]) -> Any:
    """`build()` once per session; a None result is not cached, so it is retried."""
    if key not in st.session_state:
        value = build()
        if value is None:
            return None
        st.session_state[key] = value
    return st.session_state[key]


def get_tracer() -> RunTracer:
    return _session_cached("tracer", RunTracer.from_env)


def get_vfs() -> VFS:
    return _session_cached("vfs", lambda: VFS(Settings.from_env().vfs_root))


def get_model_routers():
    """Reasoning-tier and standard-tier models, each with its own fallback; every agent is
    handed the tier it needs (see RoleModelSettings). None until a reasoning model is set."""
    return _session_cached("model_routers", build_model_routers)


def get_scope_agent():
    routers = get_model_routers()
    if routers is None:
        return None
    return _session_cached("scope_agent", lambda: build_scope_agent(routers))


def _notify_lifecycle(node: str, status: str) -> None:
    """Forward a node's lifecycle event to the chat turn that is running, if any."""
    st.session_state.get(LIFECYCLE_CALLBACK, lambda *_: None)(node, status)


def get_graph():
    """The compiled research graph with its checkpointer and run registry."""
    if "scope_graph" in st.session_state:
        return st.session_state.scope_graph
    agent = get_scope_agent()
    if agent is None:
        raise RuntimeError(
            "Reasoning-tier model is not configured. Set INTELLIGENCE_REASONING_MODEL and retry."
        )
    runtime = build_runtime(
        st.session_state.model_routers,
        agent,
        get_tracer(),
        lifecycle=_notify_lifecycle,
        warn=st.warning,
    )
    st.session_state.scope_checkpoint_connection = runtime.checkpoint_connection
    st.session_state.run_registry = runtime.run_registry
    st.session_state.scope_graph = runtime.graph
    return runtime.graph


def recovery_service() -> WorkflowRecoveryService:
    if get_scope_agent() is None:
        raise RuntimeError("Configure the reasoning model before recovering a run.")
    graph = get_graph()
    return WorkflowRecoveryService(graph, st.session_state.run_registry, get_tracer())


def inspect_checkpoint(snapshot: dict):
    """The thread's checkpoint (paused interrupt, unfinished nodes), or None before the graph
    exists or when it cannot be read."""
    if "scope_graph" not in st.session_state or "run_registry" not in st.session_state:
        return None
    service = WorkflowRecoveryService(st.session_state.scope_graph, st.session_state.run_registry)
    try:
        return service.inspect(snapshot.get("thread_id", ""))
    except Exception:
        return None
