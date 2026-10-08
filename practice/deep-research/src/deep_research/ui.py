"""Streamlit UI: chat with the agent, review the plan, review the report, publish."""

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
    "evaluate": "Evaluated the report",
    "publish": "Published the report",
}


@st.cache_resource
def get_app() -> DeepResearch:
    return DeepResearch(Settings.from_env())


def _describe(node: str, update: dict) -> str:
    if node == "supervisor":
        calls = update["supervisor_messages"][-1].tool_calls
        topics = [c["args"]["research_topic"] for c in calls if c["name"] == "ConductResearch"]
        if topics:
            return "Researching:\n" + "\n".join(f"- {t}" for t in topics)
        return "Supervisor is wrapping up research"
    return NODE_LABELS.get(node, node)


def run(start) -> None:
    """``start`` returns the turn's stream; it raises right away when the guardrail rejects input."""
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
                    status.update(label="Evaluating the report…")
        except Exception as exc:
            status.update(label="Stopped with an error", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return
        status.update(label="Done", state="complete")
    st.rerun()


def review_plan(app: DeepResearch, thread_id: str, pending: dict) -> None:
    st.subheader("Review the research plan")
    with st.form("plan"):
        topics = st.text_area("Topics, one per line (edit freely)", "\n".join(pending["topics"]), height=180)
        feedback = st.text_input("Or ask the planner for changes")
        approve = st.form_submit_button("Approve and start research", type="primary")
        revise = st.form_submit_button("Revise plan")
    if approve:
        run(lambda: app.resume(thread_id, {"approved": True, "topics": topics.splitlines()}))
    elif revise:
        if feedback.strip():
            run(lambda: app.resume(thread_id, {"approved": False, "feedback": feedback}))
        else:
            st.warning("Describe the changes you want first.")


def show_evaluation(evaluation: dict | None) -> None:
    if not evaluation:
        return
    if "error" in evaluation:
        st.caption(f"Evaluation failed: {evaluation['error']}")
    else:
        left, right = st.columns(2)
        left.metric("Faithfulness", f"{evaluation['faithfulness']:.2f}", help="Share of claims the sources support (RAGAS)")
        right.metric("Answer relevancy", f"{evaluation['answer_relevancy']:.2f}", help="How directly the report answers the brief (RAGAS)")


def review_report(app: DeepResearch, thread_id: str, pending: dict) -> None:
    st.subheader("Review the report")
    show_evaluation(pending.get("evaluation"))
    with st.container(border=True):
        st.markdown(pending["report"])
    with st.form("report"):
        feedback = st.text_area("Changes for the writer")
        approve = st.form_submit_button("Approve and publish", type="primary")
        revise = st.form_submit_button("Revise report")
    if approve:
        run(lambda: app.resume(thread_id, {"approved": True}))
    elif revise:
        if feedback.strip():
            run(lambda: app.resume(thread_id, {"approved": False, "feedback": feedback}))
        else:
            st.warning("Describe the changes you want first.")


def main() -> None:
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
        review_report(app, thread_id, view.pending)
    elif view.unfinished:
        st.warning("The last run stopped before it finished.")
        if st.button("Continue", type="primary"):
            run(lambda: app.continue_run(thread_id))
    elif view.values.get("published_path"):
        show_evaluation(view.values.get("evaluation"))
        with st.container(border=True):
            st.markdown(view.values["report"])
        st.download_button("Download report", view.values["report"], file_name="report.md")
        st.info("Start a new research from the sidebar.")
    elif prompt := st.chat_input("What should I research?"):
        with st.chat_message("user"):
            st.markdown(prompt)
        run(lambda: app.ask(thread_id, prompt))
