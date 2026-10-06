"""One chat turn: one reply drives one graph run, streamed live into the page.

A new request (or scope answer) starts or continues the research; while a plan or report awaits
review, the reply resumes the paused graph. The turn is one execution observation in the run's
Langfuse trace (see ``observability/tracing.py``); a rerun without a reply runs no turn.
"""

from __future__ import annotations

import time
from uuid import uuid4

import streamlit as st
from langgraph.types import Command

from intelligence_collector.agents.supervisor import decision_listener
from intelligence_collector.graph.nodes import Node, node_agent
from intelligence_collector.ui import resources, state
from intelligence_collector.ui.components.panel import LivePanel
from intelligence_collector.ui.html import supervisor_log_html
from intelligence_collector.ui.run_status import STOPPED, STOPPED_PROGRESS, RunView, task_numbers

# Graph node -> (activity label, tool) shown in the activity log and status block; the agent
# comes from ``NODE_AGENT``, which also names the node's Langfuse observation.
NODE_ACTIVITY = {
    Node.CLARIFY: ("Evaluating scope", None),
    Node.AWAIT_CLARIFICATION: ("Waiting for scope clarification", None),
    Node.WRITE_BRIEF: ("Generating research brief", None),
    Node.PLAN: ("Calling write_todos", "write_todos"),
    Node.AWAIT_PLAN: ("Waiting for plan approval", None),
    Node.RESEARCH: ("Delegating research to sub-agents", "tavily_search"),
    Node.ASSESS_RESEARCH: ("Assessing evidence for the research goal", "assessment"),
    Node.SYNTHESIZE: ("Synthesizing report", None),
    Node.VERIFY: ("Verifying coverage, citations and faithfulness", "report_judge"),
    Node.EVALUATE_QUALITY: ("Scoring optional RAGAS metrics", None),
    Node.FIX: ("Diagnosing the smallest repair", None),
    Node.PREPARE_REPAIR_RESEARCH: ("Preparing targeted research", None),
    Node.REPLAN: ("Proposing a replacement plan", None),
    Node.NEEDS_REVIEW: ("Escalating for review", None),
    Node.AWAIT_RECOVERY: ("Waiting for recovery decision", None),
    Node.APPROVE: ("Waiting for publication decision", None),
    Node.CLASSIFY_FEEDBACK: ("Classifying reviewer feedback", None),
    Node.PUBLISH: ("Publishing report", None),
}
EXECUTION_ID = "_execution_id"
ACTIVITY_STARTED = "_activity_started"


def record_lifecycle(node: str, status: str) -> dict:
    """Add or update this turn's activity entry for `node`, with its duration once it ends."""
    item_id = f"{st.session_state.get(EXECUTION_ID, 'execution')}:{node}"
    label, tool = NODE_ACTIVITY.get(node, (node.replace("_", " ").title(), None))
    agent = node_agent(node)
    activity = list(st.session_state.get(state.ACTIVITY, []))
    item = next((entry for entry in activity if entry.get("id") == item_id), None)
    started = st.session_state.setdefault(ACTIVITY_STARTED, {})
    if item is None:
        item = {"id": item_id, "label": label, "agent": agent, "status": status}
        if tool:
            item["tool"] = tool
        activity.append(item)
    else:
        item["status"] = status
    if status == "running":
        started[item_id] = time.monotonic()
    elif item_id in started:
        item["duration_ms"] = max(0, round((time.monotonic() - started.pop(item_id)) * 1000))
    st.session_state[state.ACTIVITY] = activity
    return item


def trace_details(text: str, view: RunView) -> tuple[str, str, dict]:
    """(execution kind, interrupt answered, observation input) of one chat turn."""
    if view.reviewing:
        return "resume", "publication_approval", {"review_reply": text}
    if view.planning:
        return "resume", "plan_approval", {"plan_reply": text}
    if view.clarifying:
        return "resume", "clarification_required", {"clarification_reply": text}
    return "start", "", {"request": text}


def stream_graph(graph, config, graph_input, panel: LivePanel) -> tuple[dict, tuple]:
    """Run the graph, redrawing the panel after every node. Returns the final state without
    LangGraph's ``__interrupt__`` marker (the stage already says where the run paused) and the
    marker itself, for the trace.

    ``stream_mode="values"`` keeps nodes on this thread, where the lifecycle callback and the
    supervisor's decision listener can reach the Streamlit session (the "custom" mode moves
    nodes onto a thread pool)."""
    result: dict = {}
    interrupted: tuple = ()
    for chunk in graph.stream(graph_input, config, stream_mode="values"):
        interrupted = chunk.get("__interrupt__") or interrupted
        result = {key: value for key, value in chunk.items() if key != "__interrupt__"}
        panel.refresh(result, st.session_state[state.ACTIVITY])
    return result, interrupted


def mark_unfinished_stopped() -> None:
    """After a turn ended early (an error, the Stop button, a closed tab): nodes of this turn
    still marked running will not finish, so they are shown as stopped, never as running."""
    execution = f"{st.session_state.get(EXECUTION_ID, 'execution')}:"
    stopped = False
    for item in st.session_state.get(state.ACTIVITY, []):
        if item.get("status") == "running" and str(item.get("id", "")).startswith(execution):
            item["status"] = STOPPED
            stopped = True
    if stopped:
        st.session_state[state.PROGRESS] = STOPPED_PROGRESS


def resume_payload(reply: state.Reply) -> dict:
    """A button's explicit decision, or the typed text for the graph to classify."""
    if reply.approved is not None:
        return {"approved": reply.approved, "actor": "local-user"}
    return {"reply": reply.text, "actor": "local-user"}


def run_chat_turn(reply: state.Reply, *, view: RunView, panel: LivePanel) -> None:
    """Answer `reply`: resume a paused review or plan, or start/continue the research. However
    the turn ends, none of its nodes is left marked as running."""
    try:
        _run_chat_turn(reply, view=view, panel=panel)
    finally:
        mark_unfinished_stopped()


def _run_chat_turn(reply: state.Reply, *, view: RunView, panel: LivePanel) -> None:
    text = reply.text
    tracer = resources.get_tracer()
    snapshot = state.run_snapshot()
    thread_id = st.session_state[state.THREAD_ID]
    st.session_state[EXECUTION_ID] = str(uuid4())
    resuming = view.reviewing or view.clarifying or view.planning
    kind, interrupt_kind, trace_input = trace_details(text, view)
    label = "Processing your reply" if resuming else "Processing research request"
    with st.status(label, expanded=False) as indicator:
        supervisor_box = st.empty()
        decisions: list[dict] = []

        def on_lifecycle(node: str, status: str) -> None:
            indicator.update(label=record_lifecycle(node, status)["label"], state="running")

        def on_supervisor_decision(entry: dict) -> None:
            decisions.append(entry)
            indicator.update(label="Research supervisor is working", expanded=True)
            supervisor_box.markdown(
                supervisor_log_html(decisions, task_numbers(snapshot)), unsafe_allow_html=True
            )

        st.session_state[resources.LIFECYCLE_CALLBACK] = on_lifecycle
        graph = resources.get_graph()
        graph_input = (
            Command(resume=resume_payload(reply))
            if resuming
            else {**snapshot, "request": text, "thread_id": thread_id}
        )
        with tracer.execution(
            run_id=snapshot["run_id"],
            thread_id=thread_id,
            kind=kind,
            interrupt_kind=interrupt_kind,
            input=trace_input,
        ) as execution:
            config = execution.graph_config(
                {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
            )
            listening = decision_listener.set(on_supervisor_decision)
            try:
                result, interrupted = stream_graph(graph, config, graph_input, panel)
            finally:
                decision_listener.reset(listening)
                st.session_state.pop(resources.LIFECYCLE_CALLBACK, None)
            execution.finish({**result, "__interrupt__": interrupted})
        indicator.update(label="Processing complete", state="complete", expanded=False)
    result["activity"] = list(st.session_state[state.ACTIVITY])
    result["trace_id"] = execution.trace_id
    state.adopt_run(result, default_progress="Scope clear; Planning started")
