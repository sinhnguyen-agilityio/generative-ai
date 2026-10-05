"""The page header: app name, run identity, stage chip and progress, the run actions (recover a
saved run, start a new thread) and the six-step stage stepper."""

from __future__ import annotations

import streamlit as st

from intelligence_collector.ui import state
from intelligence_collector.ui.components.recovery import render_checkpoint_tools
from intelligence_collector.ui.html import header_html, stage_view, stepper_html
from intelligence_collector.ui.run_status import RunView


def render_header(view: RunView) -> None:
    label, tone, steps = stage_view(
        view.stage, has_request=view.has_request, waiting_for_you=view.waiting_for_you
    )
    title_col, recover_col, new_col = st.columns([6, 1.3, 1.1], vertical_alignment="center")
    title_col.markdown(
        header_html(
            label,
            tone,
            st.session_state[state.PROGRESS],
            st.session_state[state.THREAD_ID],
            view.snapshot.get("run_id", ""),
        ),
        unsafe_allow_html=True,
    )
    # Disabled while a turn runs: a click would stop the running graph halfway.
    with recover_col.popover("Recover a run", disabled=view.busy):
        if not render_checkpoint_tools():
            st.caption("No saved runs to recover.")
    new_col.button(
        "New thread", type="primary", on_click=state.start_new_thread, disabled=view.busy
    )
    st.markdown(stepper_html(steps), unsafe_allow_html=True)
