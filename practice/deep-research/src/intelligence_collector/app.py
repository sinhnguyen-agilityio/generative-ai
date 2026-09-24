from __future__ import annotations
import os
import sqlite3
from uuid import uuid4
from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.config import Settings, load_local_env
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


def _get_scope_graph(st, agent, tracer):
    if "scope_graph" in st.session_state:
        return st.session_state.scope_graph
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = Settings.from_env()
    connection = sqlite3.connect(settings.checkpoint_path, check_same_thread=False)
    graph = build_graph(
        scope_agent=agent,
        vfs=VFS(settings.vfs_root),
        checkpointer=SqliteSaver(connection),
        tracer=tracer,
    )
    st.session_state.scope_checkpoint_connection = connection
    st.session_state.scope_graph = graph
    return graph


def render():
    load_local_env()
    try:
        import streamlit as st
    except ImportError:
        return None
    st.title("Intelligence Collector")
    st.session_state.setdefault("stage", "Scoping")
    st.session_state.setdefault("progress", "No research started")
    st.session_state.setdefault("scope_thread_id", str(uuid4()))
    st.session_state.setdefault("scope_state", {
        "run_id": str(uuid4()), "thread_id": st.session_state.scope_thread_id,
    })
    st.write(f"Stage: **{st.session_state.stage}**")
    st.write(f"Progress: {st.session_state.progress}")
    if st.session_state.get("needs_review"):
        st.error("This run needs human review before it can be treated as verified or published.")
        st.caption(st.session_state.get("review_feedback", "Unresolved verification feedback is stored with the run."))
    verification = st.session_state.get("verification")
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
                    "run_id": run_id,
                    "request": text,
                    "thread_id": thread_id,
                }
                config = {"configurable": {"thread_id": thread_id}}
                result = _get_scope_graph(st, agent, tracer).invoke(state, config)
                execution.output = {
                    "stage": result.get("stage"),
                    "scope_decision": result.get("scope_assessment", {}).get("decision"),
                }
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
