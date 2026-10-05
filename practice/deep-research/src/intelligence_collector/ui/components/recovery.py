"""Recovering runs: finding and restoring saved checkpoints, continuing unfinished work, and the
human recovery decision for a run that needs review.

These actions call the recovery service (graph work), so they run in the script body after a
button press — where errors can be shown in place — and rerun once the run was adopted.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from intelligence_collector.config import Settings
from intelligence_collector.contracts import Stage
from intelligence_collector.guardrails import normalize_text, validate_request
from intelligence_collector.storage.run_registry import RunRegistry
from intelligence_collector.ui import resources, state
from intelligence_collector.ui.run_status import THREAD_DONE_STAGES

RECOVERY_ACTIONS = ["revise", "retry_report", "research", "rescope", "export_incomplete", "cancel"]


def _adopt(result: dict, default_progress: str) -> None:
    state.adopt_run(result, default_progress=default_progress)
    st.rerun()


def render_checkpoint_tools() -> bool:
    """Find and restore saved runs (local and cheap). Returns whether anything was offered."""
    settings = Settings.from_env()
    shown = False
    if Path(settings.checkpoint_path).is_file():
        shown = True
        if st.button("Find saved checkpoints"):
            try:
                resources.recovery_service().discover_legacy_runs()
                st.rerun()
            except Exception as exc:
                st.error(f"Checkpoint discovery failed: {exc}")
    if Path(settings.run_registry_path).is_file():
        registry = st.session_state.get("run_registry") or RunRegistry(settings.run_registry_path)
        runs = registry.list_unfinished()
        if runs:
            shown = True
            labels = {
                run.thread_id: f"{run.request_summary or run.run_id} — {run.current_stage}"
                for run in runs
            }
            st.markdown("**Recover an unfinished run**")
            chosen = st.selectbox("Run", list(labels), format_func=labels.__getitem__)
            if st.button("Restore checkpoint"):
                try:
                    _adopt(
                        resources.recovery_service().load_state(chosen),
                        "Recovered checkpoint; continuation available",
                    )
                except Exception as exc:
                    st.error(f"Recovery failed: {exc}")
    return shown


def render_recovery_controls(snapshot: dict, checkpoint=None, *, disabled: bool = False) -> None:
    """While the run needs review: the bounded recovery options, right above the chat input.
    With unfinished checkpoint work: a button to continue it. Widget keys include the draft
    version, so a new draft starts with an empty form; agents are built only on an action."""
    if snapshot.get("stage") in THREAD_DONE_STAGES:
        return
    if snapshot.get("stage") == Stage.NEEDS_REVIEW.value:
        _recovery_decision(snapshot)
    elif checkpoint is not None and checkpoint.next_nodes and not checkpoint.has_interrupt:
        if st.button("Continue unfinished run", disabled=disabled):
            try:
                _adopt(
                    resources.recovery_service().continue_run(snapshot["thread_id"]),
                    "Recovered checkpoint; continuation available",
                )
            except Exception as exc:
                st.error(f"Continuation failed: {exc}")


def _recovery_decision(snapshot: dict) -> None:
    version = snapshot.get("draft_version", 0)
    # Open only the first time; after a recovery attempt returned a new draft, the draft and
    # the assistant's reply come first and the options stay collapsed until requested.
    tried = any(
        entry.get("kind") == "RECOVERY_DECISION" for entry in snapshot.get("bvf_history") or []
    )
    with st.expander("Recovery options — this run needs your decision", expanded=not tried):
        st.caption("Choose a bounded recovery action. Incomplete export is never publication.")
        action = st.selectbox(
            "Recovery action", RECOVERY_ACTIONS, key=f"recovery_action_v{version}"
        )
        feedback = st.text_area(
            "Repair instructions or research feedback",
            max_chars=4000,
            key=f"recovery_feedback_v{version}",
        )
        attempts = (
            int(
                st.number_input(
                    "Additional repair attempts",
                    min_value=1,
                    max_value=2,
                    value=1,
                    key=f"recovery_attempts_v{version}",
                )
            )
            if action == "retry_report"
            else 0
        )
        apply = st.button("Apply recovery action", key=f"recovery_apply_v{version}")
    if not apply:
        return
    try:
        if feedback and not validate_request(normalize_text(feedback)).allowed:
            raise ValueError("Recovery feedback rejected by input guardrails")
        with st.spinner("Applying the recovery action — rewriting and re-verifying the report"):
            result = resources.recovery_service().resume_interrupt(
                snapshot["thread_id"],
                {"action": action, "feedback": feedback, "additional_attempts": attempts},
            )
        _adopt(result, "Recovered checkpoint; continuation available")
    except Exception as exc:
        st.error(f"Recovery failed: {exc}")
