from __future__ import annotations
import json
import os
import sqlite3
from pathlib import Path
from uuid import uuid4
from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.config import Settings, load_local_env
from intelligence_collector.contracts import VerificationResult
from intelligence_collector.guardrails import validate_request
from intelligence_collector.graph.build import build_graph
from intelligence_collector.models.factory import ModelConfig, ModelFactory
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.observability.tracing import Tracer
from intelligence_collector.publication import can_request_approval
from intelligence_collector.storage.vfs import VFS


def _get_tracer(st):
    if "tracer" not in st.session_state:
        st.session_state.tracer = Tracer.from_env(required=True)
    return st.session_state.tracer


def _get_scope_agent(st, tracer):
    if "scope_agent" in st.session_state:
        return st.session_state.scope_agent
    model_name = os.getenv("SCOPE_MODEL", "").strip()
    if not model_name:
        return None
    provider = os.getenv("SCOPE_PROVIDER") or None
    primary = ModelFactory().create(ModelConfig(model_name, provider=provider))
    agent = ScopeAgent(ModelRouter(
        primary,
        mode="live",
        trace=tracer.model_event,
        callbacks=tracer.callbacks,
    ))
    st.session_state.scope_agent = agent
    return agent


def _open_checkpoint_connection(checkpoint_path: str):
    path = Path(checkpoint_path).expanduser()
    if path.exists() and path.is_dir():
        raise RuntimeError(
            f"Checkpoint path points to a directory, not a SQLite file: {path}"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(path, check_same_thread=False)
    except (OSError, sqlite3.Error) as exc:
        raise RuntimeError(
            f"Cannot open SQLite checkpoint database at '{path}'. "
            "Check INTELLIGENCE_CHECKPOINT_PATH and directory write permissions."
        ) from exc


def _get_scope_graph(st, agent, tracer):
    if "scope_graph" in st.session_state:
        return st.session_state.scope_graph
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = Settings.from_env()
    connection = _open_checkpoint_connection(settings.checkpoint_path)
    graph = build_graph(
        scope_agent=agent,
        vfs=VFS(settings.vfs_root),
        checkpointer=SqliteSaver(connection),
        tracer=tracer,
    )
    st.session_state.scope_checkpoint_connection = connection
    st.session_state.scope_graph = graph
    return graph


def _record_value(record, key, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _status_text(value):
    return str(getattr(value, "value", value)).replace("_", " ")


def _snapshot_verification(snapshot):
    value = snapshot.get("verification")
    if value is None:
        return None
    if isinstance(value, VerificationResult):
        return value
    try:
        return VerificationResult.model_validate(value)
    except (TypeError, ValueError):
        return None


def _render_graph_snapshot(st, snapshot):
    """Render a graph snapshot without mutating authoritative workflow data."""
    todos = list(snapshot.get("todos") or [])
    if todos:
        completed = sum(
            _status_text(_record_value(todo, "status")) == "completed"
            for todo in todos
        )
        st.subheader("Research tasks")
        st.write(f"Task progress: {completed}/{len(todos)} completed")
        for todo in todos:
            status = _status_text(_record_value(todo, "status", "unknown"))
            question = _record_value(todo, "research_question", "Untitled task")
            st.markdown(f"- **{status}** — {question}")

    verification = _snapshot_verification(snapshot)
    if verification is not None:
        st.subheader("Verification")
        if verification.passed:
            st.success("Verification: passed")
        else:
            issue = verification.issue_type.value if verification.issue_type else "unresolved"
            st.error(f"Verification: failed — {issue}")
            if verification.repair_details:
                st.write(verification.repair_details)
        if verification.checks:
            st.write("Checks: " + ", ".join(verification.checks))

    evidence = list(snapshot.get("evidence") or [])
    if evidence:
        st.subheader("Evidence")
        for item in evidence:
            title = _record_value(item, "title") or _record_value(
                item, "source_id", "Evidence"
            )
            locator = _record_value(item, "locator") or _record_value(item, "path")
            if locator:
                st.markdown(f"- [{title}]({locator})")
            else:
                st.markdown(f"- {title}")

    report = (
        snapshot.get("report")
        or snapshot.get("latest_report")
        or snapshot.get("draft")
    )
    if report:
        st.subheader("Report")
        st.markdown(report)

    return verification


def render():
    load_local_env()
    try:
        import streamlit as st
    except ImportError:
        return None
    st.title("Intelligence Collector")
    st.session_state.setdefault("stage", "Scoping")
    st.session_state.setdefault("progress", "No research started")
    existing_scope_state = st.session_state.get("scope_state", {})
    st.session_state.setdefault(
        "scope_thread_id", existing_scope_state.get("thread_id", str(uuid4()))
    )
    st.session_state.setdefault("scope_state", {
        "run_id": str(uuid4()), "thread_id": st.session_state.scope_thread_id,
    })
    st.write(f"Stage: **{st.session_state.stage}**")
    st.write(f"Progress: {st.session_state.progress}")
    snapshot_verification = _render_graph_snapshot(st, st.session_state.scope_state)
    if st.session_state.get("needs_review"):
        st.error("This run needs human review before it can be treated as verified or published.")
        st.caption(st.session_state.get("review_feedback", "Unresolved verification feedback is stored with the run."))
    verification = snapshot_verification or st.session_state.get("verification")
    if verification is not None and can_request_approval(verification):
        st.subheader("Review and publish")
        decision = st.radio("Publication decision", ["Approve", "Reject"], key="publication_decision")
        if st.button("Record decision"):
            st.session_state.approval = decision == "Approve"
            st.success("Approval recorded; publication remains hash-bound.")
    text = st.chat_input("What should we research?")
    if text:
        try:
            decision = validate_request(text)
            if not decision.allowed:
                st.error(decision.reason)
                st.session_state.progress = "Request rejected; no research started"
                return st
            tracer = _get_tracer(st)
            run_id = st.session_state.scope_state["run_id"]
            thread_id = st.session_state.scope_thread_id
            with tracer.execution(
                run_id,
                thread_id,
                input={"request": text},
            ) as execution:
                agent = _get_scope_agent(st, tracer)
                if agent is None:
                    raise RuntimeError("Scope model is not configured. Set SCOPE_MODEL and retry.")
                state = {
                    **st.session_state.scope_state,
                    "request": text,
                    "thread_id": thread_id,
                }
                config = {"configurable": {"thread_id": thread_id}}
                result = _get_scope_graph(st, agent, tracer).invoke(state, config)
                execution.output = json.dumps({
                    "stage": result.get("stage"),
                    "scope_decision": result.get("scope_assessment", {}).get("decision"),
                    "scope_assessment": result.get("scope_assessment"),
                    "research_brief": result.get("scoped_request"),
                    "messages": [
                        {"role": getattr(message, "type", "message"), "content": getattr(message, "content", str(message))}
                        for message in result.get("messages", [])
                    ],
                }, ensure_ascii=False, indent=2, default=str)
                st.session_state.scope_state = result
                st.session_state.last_trace_id = execution.trace_id
                st.session_state.stage = result.get("stage", "Scoping")
                if st.session_state.stage == "Scoping":
                    question = result.get("scope_assessment", {}).get("clarification_question")
                    st.session_state.progress = "Clarification needed"
                    st.warning(question or "Please clarify the research scope.")
                else:
                    st.session_state.progress = "Scope clear; Planning started"
                    st.success("Scope is clear. Planning started automatically.")
        except Exception as exc:
            st.error(f"Scope decision failed: {exc}")
    if st.session_state.get("last_trace_id"):
        st.caption(f"Langfuse trace: {st.session_state.last_trace_id}")
    return st
if __name__ == "__main__": render()
