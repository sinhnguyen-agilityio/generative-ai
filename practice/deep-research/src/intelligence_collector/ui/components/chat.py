"""The conversation column: messages, the plan card, the research banner, the report card, run
notices and the chat input.

Quick-reply buttons do not run anything themselves: their ``on_click`` callbacks queue the same
reply the chat input accepts ("approve", "cancel" or the requested change), and the page answers
it like a typed reply (see ``main.py``).
"""

from __future__ import annotations

import streamlit as st

from intelligence_collector.contracts import Stage
from intelligence_collector.ui import state
from intelligence_collector.ui.evidence import research_counts
from intelligence_collector.ui.html import research_banner_html, section_label_html
from intelligence_collector.ui.run_status import RESEARCH_DONE_STAGES, RunView, current_report

REVIEW_PROMPT = (
    "Reply **approve** to publish this report, or tell me what to change — I'll rewrite and "
    "re-verify it, or research again if the change needs new sources."
)
PLAN_PROMPT = "Reply **approve** to start research, **cancel** to stop, or describe what to change."


def render_conversation(view: RunView, vfs) -> None:
    """Everything above the chat input, in reading order."""
    clarification = (view.snapshot.get("scope_assessment") or {}).get("question")
    if view.stage == Stage.SCOPING.value and clarification:
        st.info("Waiting for your response")
    render_messages(view.snapshot)
    if view.planning and view.snapshot.get("plan"):
        render_plan_card(view.snapshot["plan"], disabled=view.busy)
    render_research_banner(view.snapshot, vfs)
    render_report(view.snapshot, reviewing=view.reviewing, disabled=view.busy)
    render_run_notices(view, vfs)


def render_turn_reply(text: str) -> None:
    """The reply being answered, shown until the turn's result replaces the page."""
    with st.chat_message("user"):
        st.markdown(text)


def render_messages(snapshot: dict) -> None:
    """The scope conversation and review replies."""
    for message in snapshot.get("messages") or []:
        kind = getattr(message, "type", "")
        if kind in {"human", "ai"}:
            with st.chat_message("user" if kind == "human" else "assistant"):
                st.markdown(str(message.content))


def _quick_replies(
    key: str, approve_label: str, *, can_cancel: bool = False, disabled: bool = False
) -> None:
    """Approve / Request changes (/ Cancel) under a plan or report card; disabled while a
    turn runs."""
    approve_col, change_col, cancel_col = st.columns(3)
    approve_col.button(
        approve_label,
        type="primary",
        key=f"{key}_approve",
        on_click=state.queue_reply,
        args=("approve",),
        disabled=disabled,
    )
    with change_col.popover("Request changes", disabled=disabled):
        # A form sends the change only on submit, not on every keystroke.
        with st.form(f"{key}_changes", clear_on_submit=True, border=False):
            st.text_area("What should change?", key=f"{key}_change", max_chars=4000)
            st.form_submit_button("Send", on_click=state.queue_reply_from, args=(f"{key}_change",))
    if can_cancel:
        cancel_col.button(
            "Cancel run",
            type="tertiary",
            key=f"{key}_cancel",
            on_click=state.queue_reply,
            args=("cancel",),
            disabled=disabled,
        )


def render_plan_card(plan: dict, *, disabled: bool = False) -> None:
    """The plan awaiting approval: what is asked, the numbered tasks with the evidence each
    needs, the expected report, and quick replies."""
    with st.chat_message("assistant"), st.container(border=True):
        title = (
            "Replacement research plan" if plan.get("kind") == "replacement" else "Research plan"
        )
        version = f" · hash {str(plan.get('plan_hash', ''))[:6]}" if plan.get("plan_hash") else ""
        st.markdown(f"**{title}**{version}")
        st.caption("Nothing is searched until you approve this exact plan.")
        if plan.get("required_answers"):
            st.markdown(section_label_html("Required answers"), unsafe_allow_html=True)
            st.markdown("\n".join(f"- {item}" for item in plan["required_answers"]))
        st.markdown(section_label_html("Research tasks"), unsafe_allow_html=True)
        for number, task in enumerate(plan.get("tasks", []), start=1):
            st.markdown(f"**{number}.** {task['question']}")
            if task.get("evidence_needed"):
                st.caption(f"Evidence: {task['evidence_needed']}")
        if plan.get("report_shape"):
            st.markdown(f"**Report shape:** {plan['report_shape']}")
        st.caption(PLAN_PROMPT)
        _quick_replies("plan", "Approve plan", can_cancel=True, disabled=disabled)


def render_research_banner(snapshot: dict, vfs) -> None:
    """Once research has finished: tasks, sources and supervisor follow-ups in one line."""
    todos = list(snapshot.get("todos") or [])
    run_id = snapshot.get("run_id", "")
    if snapshot.get("stage") not in RESEARCH_DONE_STAGES or not todos or not run_id:
        return
    counts = research_counts(vfs, run_id, todos)
    st.markdown(
        research_banner_html(counts.tasks, counts.sources, counts.follow_ups),
        unsafe_allow_html=True,
    )


def render_report(snapshot: dict, *, reviewing: bool, disabled: bool = False) -> None:
    """The current report: in review with a download and quick replies; published or
    unverified drafts as they are."""
    report = current_report(snapshot)
    if not report:
        return
    if reviewing:
        with st.chat_message("assistant"):
            st.markdown(report)
            st.caption(REVIEW_PROMPT)
            st.download_button(
                "Download .md",
                report,
                file_name=f"report-v{snapshot.get('draft_version', 1)}.md",
                mime="text/markdown",
                key="report_download",
            )
            _quick_replies("report", "Approve and publish", disabled=disabled)
    elif snapshot.get("stage") == Stage.PUBLISHED.value:
        with st.chat_message("assistant"):
            st.markdown(report)
    elif snapshot.get("stage") == Stage.NEEDS_REVIEW.value:
        with st.chat_message("assistant"):
            version = snapshot.get("draft_version", 1)
            st.caption(f"Unverified draft v{version} — not approved or published")
            st.markdown(report)


def render_run_notices(view: RunView, vfs) -> None:
    """Review, publication, unfinished-work and incomplete-export notices."""
    snapshot = view.snapshot
    if view.run_stage == Stage.NEEDS_REVIEW.value:
        st.error("This run needs human review before it can be treated as verified or published.")
        st.caption(
            snapshot.get("current_action")
            or snapshot.get("last_error")
            or "Unresolved verification feedback is stored with the run."
        )
    elif view.run_stage == Stage.PUBLISHED.value:
        destination = (snapshot.get("publication") or {}).get(
            "destination", "(unknown destination)"
        )
        st.success(f"Published to {destination}")
    if view.pending:
        st.info("This checkpoint has unfinished work. Use Continue unfinished run to resume it.")
    if view.run_stage in {"Exported Incomplete", Stage.INCOMPLETE.value}:
        st.warning(
            "Incomplete answer — not a published report. Some requested information could not "
            "be verified."
        )
        try:
            st.download_button(
                "Download incomplete report",
                vfs.read(snapshot["run_id"], "reports/incomplete.md"),
                file_name="incomplete-report.md",
                mime="text/markdown",
            )
        except FileNotFoundError:
            st.error("The incomplete report artifact is unavailable.")


def chat_input(view: RunView) -> str | None:
    """The reply typed this rerun, or a queued quick reply."""
    typed = st.chat_input(view.input_placeholder, disabled=view.input_disabled)
    return state.take_pending_reply() or typed
