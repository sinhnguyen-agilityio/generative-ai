"""Streamlit UI: chat with the agent, review the plan, read the report, follow up or publish."""

from __future__ import annotations

import uuid

import streamlit as st

from deep_research.config import Settings
from deep_research.agents.guardrail import GuardrailViolation
from deep_research.runner import DeepResearch

NODE_LABELS = {
    "clarify": "Understood the request",
    "write_brief": "Wrote the research brief",
    "planner": "Drafted the research plan",
    "supervisor_tools": "Researchers finished a round",
    "write_report": "Wrote the report",
    "verify": "Checked the report against its sources",
    "triage": "Read your follow-up",
    "evaluate": "Evaluated the report",
    "publish": "Published the report",
}


@st.cache_resource
def get_app() -> DeepResearch:
    """Create and cache the application runner for Streamlit sessions.

    Returns:
        DeepResearch runner configured from the environment.
    """
    return DeepResearch(Settings.from_env())


def _describe(node: str, update: dict) -> str:
    """Describe a completed node or the topics delegated by the supervisor.

    Args:
        node: Graph node emitting the update.
        update: Node state update, including supervisor tool calls when applicable.

    Returns:
        Human-readable progress message.
    """
    if node == "supervisor":
        calls = update["supervisor_messages"][-1].tool_calls
        topics = [c["args"]["research_topic"] for c in calls if c["name"] == "ConductResearch"]
        if topics:
            return "Researching:\n" + "\n".join(f"- {t}" for t in topics)
        return "Supervisor is wrapping up research"
    return NODE_LABELS.get(node, node)


def run(start) -> None:
    """Display graph progress and errors, then rerun the UI after a successful turn.

    Args:
        start: Callable returning a node-update iterator; input rejection is shown immediately.
    """
    try:
        stream = start()
    except GuardrailViolation as exc:
        st.error(f"Message blocked: {exc}")
        return
    with st.status("Working…", expanded=True) as status:
        try:
            for node, update in stream:
                status.write(_describe(node, update))
                if node == "write_report":
                    status.update(label="Checking the report…")
        except Exception as exc:
            status.update(label="Stopped with an error", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return
        status.update(label="Done", state="complete")
    st.rerun()


def review_plan(app: DeepResearch, thread_id: str, pending: dict) -> None:
    """Render editable topics and criteria, then submit approval or revision feedback.

    Args:
        app: Runner used to resume the review interrupt.
        thread_id: Thread waiting for plan review.
        pending: Interrupt payload containing topics and acceptance criteria.
    """
    st.subheader("Review the research plan")
    with st.form("plan"):
        topics = st.text_area("Topics, one per line (edit freely)", "\n".join(pending["topics"]), height=180)
        criteria = st.text_area(
            "Acceptance criteria: what the report must answer, one per line",
            "\n".join(pending.get("criteria", [])),
            height=120,
        )
        feedback = st.text_input("Or ask the planner for changes")
        approve = st.form_submit_button("Approve and start research", type="primary")
        revise = st.form_submit_button("Revise plan")
    if approve:
        run(lambda: app.resume(thread_id, {"approved": True, "topics": topics.splitlines(), "criteria": criteria.splitlines()}))
    elif revise:
        if feedback.strip():
            run(lambda: app.resume(thread_id, {"approved": False, "feedback": feedback}))
        else:
            st.warning("Describe the changes you want first.")


def show_evaluation(evaluation: dict | None) -> None:
    """Display evaluation metrics or a recorded evaluation error when present.

    Args:
        evaluation: Evaluation scores, error dictionary, or None.
    """
    if not evaluation:
        return
    if "error" in evaluation:
        st.caption(f"Evaluation failed: {evaluation['error']}")
    else:
        left, right = st.columns(2)
        left.metric("Faithfulness", f"{evaluation['faithfulness']:.2f}", help="Share of claims the sources support (RAGAS)")
        right.metric("Answer relevancy", f"{evaluation['answer_relevancy']:.2f}", help="How directly the report answers the brief (RAGAS)")


def show_verification(verification: dict | None) -> None:
    """Display citation support, removed sentences, and unmet criteria when present.

    Args:
        verification: Final verification summary, error dictionary, or None.
    """
    if not verification:
        return
    if "error" in verification:
        st.caption(f"Source check failed: {verification['error']}")
        return
    st.caption(
        f"Source check: {verification['support']:.0%} of claims supported by their cited sources "
        f"after {verification['rounds']} fix round(s); {len(verification['removed'])} unsupported sentence(s) removed."
    )
    if verification["unmet"]:
        st.warning(
            "Not answered by the research so far (ask for more research below):\n"
            + "\n".join(f"- {c}" for c in verification["unmet"])
        )


def answer(app: DeepResearch, thread_id: str, pending: dict) -> None:
    """Display the reviewed report and accept publication or follow-up requests.

    Args:
        app: Runner used to submit the review decision.
        thread_id: Thread waiting for report review.
        pending: Report interrupt payload with optional verification and evaluation.
    """
    with st.chat_message("assistant"):
        st.markdown(pending["report"])
        show_verification(pending.get("verification"))
        show_evaluation(pending.get("evaluation"))
        if st.button("Publish report", type="primary"):
            run(lambda: app.resume(thread_id, {"approved": True}))
    if prompt := st.chat_input("Ask for more research or changes to the report"):
        with st.chat_message("user"):
            st.markdown(prompt)
        run(lambda: app.resume(thread_id, {"approved": False, "feedback": prompt}))


def main() -> None:
    """Render the research UI and select the current chat, review, or recovery screen."""
    st.set_page_config(page_title="Deep Research", layout="wide")
    app = get_app()
    thread_id = st.session_state.setdefault("thread_id", str(uuid.uuid4()))

    with st.sidebar:
        st.title("Deep Research")
        if st.button("New research", width="stretch"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.rerun()
        st.caption(f"Thread `{thread_id[:8]}`")
        st.caption(f"Langfuse tracing: {'on' if app.settings.langfuse_enabled else 'off'}")

    view = app.view(thread_id)
    for message in view.values.get("messages", []):
        with st.chat_message("user" if message.type == "human" else "assistant"):
            st.markdown(message.text)

    if view.pending and view.pending["type"] == "plan":
        review_plan(app, thread_id, view.pending)
    elif view.pending and view.pending["type"] == "report":
        answer(app, thread_id, view.pending)
    elif view.unfinished:
        st.warning("The last run stopped before it finished.")
        if st.button("Continue", type="primary"):
            run(lambda: app.continue_run(thread_id))
    elif view.values.get("published_path"):
        show_verification(view.values.get("verification"))
        show_evaluation(view.values.get("evaluation"))
        with st.container(border=True):
            st.markdown(view.values["report"])
        st.download_button("Download report", view.values["report"], file_name="report.md")
        st.info("Start a new research from the sidebar.")
    elif prompt := st.chat_input("What should I research?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        run(lambda: app.ask(thread_id, prompt))
