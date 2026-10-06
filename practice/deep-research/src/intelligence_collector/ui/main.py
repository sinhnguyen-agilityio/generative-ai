"""The page: header, conversation and side panel, then the answer to the user's reply.

Streamlit runs this top to bottom on every interaction. ``main`` only composes components;
state lives in ``state.py``, dependencies in ``resources.py``, the graph turn in ``turn.py``.

A reply is answered in two reruns. The first checks it with the input guardrails and accepts
it; the second renders the page with every input disabled and runs the graph turn. A click
during a turn would request a rerun, which stops the running script mid-node — so nothing is
clickable while a turn runs.
"""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from intelligence_collector.config import load_local_env
from intelligence_collector.guardrails import GuardrailViolation, normalize_text, validate_request
from intelligence_collector.ui import resources, state, turn
from intelligence_collector.ui.components.chat import (
    chat_input,
    render_conversation,
    render_turn_reply,
)
from intelligence_collector.ui.components.header import render_header
from intelligence_collector.ui.components.panel import LivePanel, render_panel
from intelligence_collector.ui.components.recovery import render_recovery_controls
from intelligence_collector.ui.html import theme_css
from intelligence_collector.ui.run_status import RunView, failure_stage_label


def main() -> None:
    load_local_env()
    st.set_page_config(page_title="Intelligence Collector", layout="wide")
    st.markdown(theme_css(), unsafe_allow_html=True)
    state.init_session()

    turn_reply = state.take_turn_reply()  # accepted in the previous rerun; runs now
    snapshot = state.run_snapshot()
    checkpoint = resources.inspect_checkpoint(snapshot)
    view = RunView.build(
        st.session_state[state.STAGE], snapshot, checkpoint, busy=turn_reply is not None
    )
    vfs = resources.get_vfs()

    render_header(view)
    tracing_problem = resources.get_tracer().disabled_reason
    if tracing_problem:
        st.caption(f"Langfuse tracing is off ({tracing_problem}); research runs normally.")

    chat_col, panel_col = st.columns([3, 2], gap="large")
    with panel_col:
        panel = render_panel(snapshot, vfs)
    with chat_col:
        render_conversation(view, vfs)
        render_recovery_controls(snapshot, checkpoint, disabled=view.busy)
        if turn_reply is not None:
            render_turn_reply(turn_reply.text)
            _run_turn(turn_reply, view, panel)
            # Still here: the turn failed and its error is shown. Let the user reply again.
            view = replace(view, busy=False)
        reply = chat_input(view)
        if reply and not view.busy:
            _accept(reply, view)


def _accept(reply: state.Reply, view: RunView) -> None:
    """Check the reply with the input guardrails; accepted, its turn runs in the next rerun."""
    text = normalize_text(reply.text)
    try:
        decision = validate_request(text)
    except GuardrailViolation as exc:
        st.error(f"Request blocked: {exc}")
        return
    if not decision.allowed:
        st.error(decision.reason)
        if not view.reviewing:
            st.session_state[state.PROGRESS] = "Request rejected; no research started"
        return
    state.start_turn(reply._replace(text=text))
    st.rerun()


def _run_turn(reply: state.Reply, view: RunView, panel: LivePanel) -> None:
    """Run one graph turn for an accepted reply, then show its result."""
    try:
        turn.run_chat_turn(reply, view=view, panel=panel)
    except GuardrailViolation as exc:
        _mark_last_activity_failed()
        st.error(f"Request blocked: {exc}")
        return
    except Exception as exc:
        # A turn spans many nodes (Scope through Publish): name the one that was running.
        label = failure_stage_label(state.activity())
        _mark_last_activity_failed()
        st.error(f"{label} failed: {exc}")
        return
    st.rerun()


def _mark_last_activity_failed() -> None:
    if state.activity():
        state.activity()[-1]["status"] = "failure"
