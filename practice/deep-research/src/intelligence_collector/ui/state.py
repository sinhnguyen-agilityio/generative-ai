"""The UI's session state: one place for its keys, their defaults and every change to them.

Components read state through these helpers and change it only through the functions below
(most are ``on_click`` callbacks), so a widget never has to call ``st.rerun()`` just to update
state.
"""

from __future__ import annotations

from typing import NamedTuple
from uuid import uuid4

import streamlit as st

from intelligence_collector.ui.run_status import PROGRESS_BY_STAGE

# ----- keys -----
STAGE = "stage"  # the stage shown in the header
PROGRESS = "progress"  # the one-line progress text next to it
ACTIVITY = "activity"  # node lifecycle log of the current thread
THREAD_ID = "scope_thread_id"
RUN = "scope_state"  # the latest graph state of the current run
PENDING_REPLY = "pending_reply"  # a quick-reply button's reply, consumed by the next rerun
TURN_REPLY = "turn_reply"  # an accepted reply whose graph turn runs in the next rerun

NO_RESEARCH = "No research started"


class Reply(NamedTuple):
    """One chat turn's input: the text shown in the conversation and, for an Approve / Cancel
    button, its explicit decision (``None`` for typed text, which the graph classifies)."""

    text: str
    approved: bool | None = None


def init_session() -> None:
    """Create the defaults once per browser session; later reruns keep what is there."""
    st.session_state.setdefault(STAGE, "Scoping")
    st.session_state.setdefault(PROGRESS, NO_RESEARCH)
    st.session_state.setdefault(ACTIVITY, [])
    existing = st.session_state.get(RUN, {})
    st.session_state.setdefault(THREAD_ID, existing.get("thread_id", str(uuid4())))
    st.session_state.setdefault(
        RUN, {"run_id": str(uuid4()), "thread_id": st.session_state[THREAD_ID]}
    )


def run_snapshot() -> dict:
    return st.session_state[RUN]


def activity() -> list:
    return st.session_state[ACTIVITY]


def start_new_thread() -> None:
    """``on_click``: reset conversation-scoped state only; cached models, graph and tracer stay.
    Nothing is open in Langfuse between turns, so there is nothing to close."""
    thread_id = str(uuid4())
    st.session_state[THREAD_ID] = thread_id
    st.session_state[RUN] = {"run_id": str(uuid4()), "thread_id": thread_id}
    st.session_state[ACTIVITY] = []
    st.session_state[STAGE] = "Scoping"
    st.session_state[PROGRESS] = NO_RESEARCH
    st.session_state.pop(PENDING_REPLY, None)
    st.session_state.pop(TURN_REPLY, None)


def adopt_run(result: dict, *, default_progress: str) -> None:
    """Make `result` (a finished turn or a recovered checkpoint) the current run."""
    st.session_state[RUN] = result
    st.session_state[THREAD_ID] = result.get("thread_id", st.session_state.get(THREAD_ID))
    st.session_state[STAGE] = result.get("stage", "Scoping")
    st.session_state[ACTIVITY] = list(result.get("activity", []))
    st.session_state[PROGRESS] = PROGRESS_BY_STAGE.get(st.session_state[STAGE], default_progress)


def queue_reply(reply: str, approved: bool) -> None:
    """``on_click`` of an Approve / Cancel button: answer with an explicit decision."""
    st.session_state[PENDING_REPLY] = Reply(reply, approved)


def queue_reply_from(widget_key: str) -> None:
    """``on_click`` of a form's submit button: queue the text typed in `widget_key`."""
    text = str(st.session_state.get(widget_key) or "").strip()
    if text:
        st.session_state[PENDING_REPLY] = Reply(text)


def take_pending_reply() -> Reply | None:
    """The queued quick reply, removed so it is answered exactly once."""
    return st.session_state.pop(PENDING_REPLY, None)


def start_turn(reply: Reply) -> None:
    """Accept `reply`: its graph turn runs in the next rerun, which renders every input disabled,
    so no click can interrupt the turn halfway (a rerun request stops a running script)."""
    st.session_state[TURN_REPLY] = reply


def take_turn_reply() -> Reply | None:
    """The accepted reply whose turn runs in this rerun, removed so it runs exactly once."""
    return st.session_state.pop(TURN_REPLY, None)
