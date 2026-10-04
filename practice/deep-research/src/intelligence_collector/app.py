"""Streamlit UI: the chat on the left and a framed Pipeline / Activity log / Details panel on the
right. Each chat message drives one graph turn; while a report awaits review, the message is the
reviewer's reply.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from uuid import uuid4
from langgraph.types import Command
from intelligence_collector.config import Settings, load_local_env
from intelligence_collector.contracts import Stage, VerificationResult
from intelligence_collector.guardrails import GuardrailViolation, normalize_text, validate_request
from intelligence_collector.graph.recovery import WorkflowRecoveryService
from intelligence_collector.graph.nodes import Node
from intelligence_collector.observability.tracing import RunTracer
from intelligence_collector.publication import can_request_approval
from intelligence_collector.runtime import (
    build_model_routers,
    build_runtime,
    build_scope_agent,
)
from intelligence_collector.storage.run_registry import RunRegistry
from intelligence_collector.storage.vfs import VFS


# ===== SESSION-CACHED DEPENDENCIES (built once per browser session, see runtime.py) =====


def _get_tracer(st):
    if "tracer" not in st.session_state:
        st.session_state.tracer = RunTracer.from_env()
    return st.session_state.tracer


def _get_model_routers(st):
    """Reasoning-tier and standard-tier models, each with its own fallback; every agent is
    handed the tier it needs (see RoleModelSettings). Model calls are traced through the graph
    execution they run in, so the routers carry no tracing dependency."""
    if "model_routers" in st.session_state:
        return st.session_state.model_routers
    routers = build_model_routers()
    if routers is not None:
        st.session_state.model_routers = routers
    return routers


def _get_scope_agent(st):
    if "scope_agent" in st.session_state:
        return st.session_state.scope_agent
    routers = _get_model_routers(st)
    if routers is None:
        return None
    st.session_state.scope_agent = build_scope_agent(routers)
    return st.session_state.scope_agent


def _get_scope_graph(st, agent, tracer):
    if "scope_graph" in st.session_state:
        return st.session_state.scope_graph
    runtime = build_runtime(
        st.session_state.model_routers,
        agent,
        tracer,
        lifecycle=lambda node, status: st.session_state.get("_lifecycle_callback", lambda *_: None)(
            node, status
        ),
        warn=st.warning,
    )
    st.session_state.scope_checkpoint_connection = runtime.checkpoint_connection
    st.session_state.run_registry = runtime.run_registry
    st.session_state.scope_graph = runtime.graph
    return runtime.graph


def _get_vfs(st):
    if "vfs" not in st.session_state:
        st.session_state.vfs = VFS(Settings.from_env().vfs_root)
    return st.session_state.vfs


def _start_new_thread(st):
    """Reset conversation-scoped state only; cached model/graph/tracer connections stay put.
    Nothing is open in Langfuse between turns, so there is nothing to close here."""
    new_thread_id = str(uuid4())
    st.session_state.scope_thread_id = new_thread_id
    st.session_state.scope_state = {"run_id": str(uuid4()), "thread_id": new_thread_id}
    st.session_state.activity = []
    st.session_state.stage = "Scoping"
    st.session_state.progress = "No research started"
    for key in ("verification",):
        st.session_state.pop(key, None)


_QUALITY_METRICS = (
    ("faithfulness", "Faithfulness"),
    ("context_precision", "Context precision"),
    ("response_relevancy", "Response relevancy"),
)


def _render_quality_scores(st, snapshot):
    """RAGAS scores of the final report side by side; informational, never a gate."""
    present = [
        (key, label)
        for key, label in _QUALITY_METRICS
        if snapshot.get(f"{key}_score") is not None or snapshot.get(f"{key}_error")
    ]
    if not present:
        return
    st.subheader("Report quality (RAGAS)")
    for column, (key, label) in zip(st.columns(len(present)), present):
        score = snapshot.get(f"{key}_score")
        with column:
            if score is not None:
                st.metric(label, f"{score:.0%}")
            else:
                st.caption(f"{label} unavailable: {snapshot.get(key + '_error')}")
    faithfulness = snapshot.get("faithfulness_score")
    if faithfulness is not None and faithfulness < 0.9:
        st.warning("Below the 90% faithfulness target.")


def _record_value(record, key, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _status_text(value):
    return str(getattr(value, "value", value)).replace("_", " ")


def _failure_stage_label(activity: list) -> str:
    """Name the node that was actually running when a turn-wide exception was caught, instead of
    always blaming whichever node happens to be first in that turn's try block."""
    if not activity:
        return "Request"
    return activity[-1].get("label") or "Request"


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


def _render_details_content(st, snapshot):
    """Render a graph snapshot without mutating authoritative workflow data."""
    current_action = snapshot.get("current_action")
    if current_action:
        st.write(f"Current action: {current_action}")

    assessment = snapshot.get("progress_assessment")
    if assessment:
        st.subheader("Supervisor assessment")
        next_action = _record_value(assessment, "next_action", "")
        blockers = list(_record_value(assessment, "blockers", []) or [])
        if next_action:
            st.write(f"Next action: {next_action}")
        if blockers:
            st.write("Blockers: " + ", ".join(blockers))

    attempt = int(snapshot.get("recovery_attempt") or 0)
    if attempt:
        st.subheader("Improvement cycles")
        st.write(f"Cycle: {attempt}/{snapshot.get('recovery_budget', 0)}")
        for entry in snapshot.get("bvf_history") or []:
            if entry.get("route"):
                st.markdown(
                    f"- Cycle {entry.get('attempt')}: {entry['route']} repair of "
                    + ", ".join(entry.get("failed_checks") or [])
                )

    todos = list(snapshot.get("todos") or [])
    if todos:
        completed = sum(
            _status_text(_record_value(todo, "status")) == "completed" for todo in todos
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
            st.error("Verification: failed")
            if verification.repair_details:
                st.write(verification.repair_details)
        if verification.checks:
            st.write("Checks: " + ", ".join(verification.checks))

    _render_quality_scores(st, snapshot)

    evidence = list(snapshot.get("evidence") or [])
    if evidence:
        st.subheader("Evidence")
        for item in evidence:
            title = _record_value(item, "title") or _record_value(item, "source_id", "Evidence")
            locator = _record_value(item, "locator") or _record_value(item, "path")
            if locator:
                st.markdown(f"- [{title}]({locator})")
            else:
                st.markdown(f"- {title}")

    report = snapshot.get("report") or snapshot.get("latest_report") or snapshot.get("draft")
    if report:
        st.subheader("Report")
        st.markdown(report)

    return verification


_AGENT_ROWS = [
    ("Scope Agent", "Scope Agent"),
    ("Planner", "Planner"),
    ("Researcher", "Research Supervisor"),
    ("Synthesizer", "Report Writer"),
    ("Verifier", "Verifier"),
]

_STATUS_BADGE = {
    "pending": "⚪ Pending",
    "running": "⏳ Running",
    "complete": "✅ Complete",
    "failed": "❌ Failed",
}


def _agent_status(activity, agent_name):
    items = [item for item in activity if _record_value(item, "agent") == agent_name]
    if not items:
        return "pending"
    if any(_record_value(item, "status") == "running" for item in items):
        return "running"
    if any(_record_value(item, "status") == "failure" for item in items):
        return "failed"
    return "complete"


def _scope_agent_status(snapshot, activity):
    """A successful clarify_with_user turn only means one question was asked, not that the
    Scope Agent is done — that only happens once a research brief actually exists."""
    if snapshot.get("scoped_request"):
        return "complete"
    base = _agent_status(activity, "Scope Agent")
    return "running" if base in {"running", "complete"} else base


def _agent_row_status(snapshot, activity, agent_key):
    if agent_key == "Scope Agent":
        return _scope_agent_status(snapshot, activity)
    return _agent_status(activity, agent_key)


def _todo_sources(vfs, run_id: str, todo_id: str) -> list[dict]:
    """Tavily sources captured for one completed todo, read from its evidence bundle."""
    try:
        bundle = json.loads(vfs.read(run_id, f"research/{todo_id}/evidence/result.json"))
    except FileNotFoundError:
        return []
    seen: set[str] = set()
    sources = []
    for source in bundle.get("sources", []):
        locator = source.get("locator")
        if not locator or locator in seen:
            continue
        seen.add(locator)
        title = (source.get("provenance") or {}).get("title") or locator
        sources.append({"locator": locator, "title": title})
    return sources


def _render_agent_row_body(st, agent_key, status, snapshot, vfs):
    """The agent-specific content shown under its title: a processing note while running,
    otherwise its key output once complete."""
    if agent_key == "Scope Agent":
        if status == "running":
            st.caption("Processing your request…")
        elif status == "complete":
            brief = (snapshot.get("scoped_request") or {}).get("research_question")
            if brief:
                st.write(brief)
    elif agent_key == "Planner":
        if status == "running":
            st.caption("Decomposing the brief into research questions…")
        elif status == "complete":
            for todo in snapshot.get("todos") or []:
                st.markdown(f"- {_record_value(todo, 'research_question', 'Untitled task')}")
    elif agent_key == "Research Supervisor":
        if status == "running":
            st.caption("Searching sources for each question…")
        elif status == "complete":
            run_id = snapshot.get("run_id", "")
            completed_todos = [
                todo
                for todo in snapshot.get("todos") or []
                if _status_text(_record_value(todo, "status")) == "completed"
            ]
            for todo in completed_todos:
                question = _record_value(todo, "research_question", "Untitled task")
                todo_id = _record_value(todo, "todo_id", "")
                sources = _todo_sources(vfs, run_id, todo_id) if todo_id and run_id else []
                with st.expander(question, expanded=False):
                    if sources:
                        for source in sources:
                            st.markdown(f"- [{source['title']}]({source['locator']})")
                    else:
                        st.caption("No sources recorded.")
    elif agent_key == "Report Writer":
        if status == "running":
            st.caption("Drafting the report…")
        elif status == "complete":
            draft = snapshot.get("draft")
            if draft:
                st.caption(f"{len(draft.split())} words drafted — full report and checks below.")
    elif agent_key == "Verifier":
        if status == "running":
            st.caption("Judging the report against the evidence…")
        elif status == "complete":
            verification = _snapshot_verification(snapshot)
            if verification is not None:
                if verification.passed:
                    st.success("All checks passed")
                else:
                    failed = [c.name for c in verification.check_results if not c.passed]
                    st.error(f"Failed — {', '.join(failed)}")


def _render_agent_row(st, placeholder, label, agent_key, snapshot, vfs):
    """One agent per row: title/content on the left, its running/complete badge on the right
    of that same row — redrawn into `placeholder` so it can be updated live during streaming.
    A row stays hidden until that agent actually starts processing — the pipeline shouldn't
    list every agent as "Pending" before the user has even sent a request."""
    activity = list(snapshot.get("activity") or [])
    status = _agent_row_status(snapshot, activity, agent_key)
    if status == "pending":
        placeholder.empty()
        return
    with placeholder.container():
        content_col, badge_col = st.columns([5, 1])
        with content_col:
            st.markdown(f"**{label}**")
            _render_agent_row_body(st, agent_key, status, snapshot, vfs)
        with badge_col:
            st.markdown(_STATUS_BADGE[status])


def _render_agent_pipeline(st, placeholders, snapshot, vfs):
    for label, agent_key in _AGENT_ROWS:
        _render_agent_row(st, placeholders[agent_key], label, agent_key, snapshot, vfs)


def _render_activity_log(st, placeholder, activity):
    """Chronological node log, redrawn into `placeholder` so it updates live while streaming."""
    with placeholder.container():
        if not activity:
            st.caption("No activity yet.")
            return
        for item in activity:
            label = _record_value(item, "label", "Workflow operation")
            status = _status_text(_record_value(item, "status", "unknown"))
            agent = _record_value(item, "agent", "Workflow")
            tool = _record_value(item, "tool")
            duration = _record_value(item, "duration_ms")
            details = [agent]
            if tool:
                details.append(f"tool: {tool}")
            if duration is not None:
                details.append(f"{duration} ms")
            st.markdown(f"- **{status}** — {label} · {' · '.join(details)}")


def _render_graph_snapshot(st, snapshot):
    """Keep workflow internals available without competing with the chat."""
    with st.expander("Details", expanded=False):
        verification = _render_details_content(st, snapshot)
        if snapshot.get("trace_id"):
            st.caption(f"Langfuse trace: {snapshot['trace_id']}")
    return verification


_REVIEW_PROMPT = (
    "Reply **approve** to publish this report, or tell me what to change — I'll rewrite and "
    "re-verify it, or research again if the change needs new sources."
)


def _awaiting_review(snapshot) -> bool:
    """Gated on the live stage, not just the (immutable) verification result: once published or
    routed back into the pipeline, the chat must stop treating replies as a review."""
    verification = _snapshot_verification(snapshot)
    return (
        snapshot.get("stage") == Stage.AWAITING_APPROVAL.value
        and verification is not None
        and can_request_approval(verification)
    )


def _awaiting_plan_approval(checkpoint) -> bool:
    """Only the checkpoint knows whether a plan is paused for approval (scope-only runs also
    end at the Planning stage, without an interrupt)."""
    return checkpoint is not None and checkpoint.interrupt_kind == "plan_approval"


def _render_plan(st, plan: dict):
    """The plan awaiting approval: what is asked, the research tasks and the expected report."""
    with st.chat_message("assistant"):
        title = (
            "Replacement research plan" if plan.get("kind") == "replacement" else "Research plan"
        )
        lines = [
            f"**{title}** — nothing is searched until you approve it.",
            "",
            "Required answers:",
            *(f"- {item}" for item in plan.get("required_answers", [])),
            "",
            "Research tasks:",
        ]
        for task in plan.get("tasks", []):
            needed = (
                f" _(evidence: {task['evidence_needed']})_" if task.get("evidence_needed") else ""
            )
            lines.append(f"- {task['question']}{needed}")
        if plan.get("report_shape"):
            lines += ["", f"Report: {plan['report_shape']}"]
        st.markdown("\n".join(lines))
        st.caption(
            "Reply **approve** to start research, **cancel** to stop, or describe what to change."
        )


def _awaiting_clarification(snapshot) -> bool:
    assessment = snapshot.get("scope_assessment") or {}
    return (
        snapshot.get("stage") == Stage.SCOPING.value
        and assessment.get("need_clarification") is True
        and bool(str(assessment.get("question") or "").strip())
    )


def _awaiting_recovery(snapshot) -> bool:
    return snapshot.get("stage") == Stage.NEEDS_REVIEW.value


def _render_transcript(st, snapshot):
    """The chat: the scope conversation and review replies, then the current report — which,
    while awaiting review, asks the user to approve or request changes by replying."""
    for message in list(snapshot.get("messages") or []):
        kind = getattr(message, "type", "")
        if kind not in {"human", "ai"}:
            continue
        with st.chat_message("user" if kind == "human" else "assistant"):
            st.markdown(str(message.content))
    report = snapshot.get("report") or snapshot.get("latest_report") or snapshot.get("draft")
    if not report:
        return
    if _awaiting_review(snapshot):
        with st.chat_message("assistant"):
            st.markdown(report)
            st.caption(_REVIEW_PROMPT)
    elif snapshot.get("stage") == Stage.PUBLISHED.value:
        with st.chat_message("assistant"):
            st.markdown(report)
    elif _awaiting_recovery(snapshot):
        with st.chat_message("assistant"):
            st.caption(
                f"Unverified draft v{snapshot.get('draft_version', 1)} — not approved or published"
            )
            st.markdown(report)


_NODE_ACTIVITY = {
    Node.CLARIFY: ("Evaluating scope", "Scope Agent", None),
    Node.AWAIT_CLARIFICATION: ("Waiting for scope clarification", "Scope Agent", None),
    Node.WRITE_BRIEF: ("Generating research brief", "Scope Agent", None),
    Node.PLAN: ("Calling write_todos", "Planner", "write_todos"),
    Node.AWAIT_PLAN: ("Waiting for plan approval", "Planner", None),
    Node.RESEARCH: ("Delegating research to sub-agents", "Research Supervisor", "tavily_search"),
    Node.ASSESS_RESEARCH: (
        "Assessing evidence for the research goal",
        "Research Evidence Judge",
        "assessment",
    ),
    Node.SYNTHESIZE: ("Synthesizing report", "Report Writer", None),
    Node.VERIFY: ("Verifying coverage, citations and faithfulness", "Verifier", "report_judge"),
    Node.EVALUATE_QUALITY: ("Scoring optional RAGAS metrics", "Verifier", None),
    Node.FIX: ("Diagnosing the smallest repair", "Repair Diagnosis", None),
    Node.PREPARE_REPAIR_RESEARCH: ("Preparing targeted research", "Research Supervisor", None),
    Node.REPLAN: ("Proposing a replacement plan", "Planner", None),
    Node.NEEDS_REVIEW: ("Escalating for review", "Verifier", None),
    Node.AWAIT_RECOVERY: ("Waiting for recovery decision", "Verifier", None),
    Node.APPROVE: ("Waiting for publication decision", "Publisher", None),
    Node.CLASSIFY_FEEDBACK: ("Classifying reviewer feedback", "Feedback Router", None),
    Node.PUBLISH: ("Publishing report", "Publisher", None),
}


def _record_lifecycle(st, node: str, status: str):
    execution_id = st.session_state.get("_execution_id", "execution")
    item_id = f"{execution_id}:{node}"
    label, agent, tool = _NODE_ACTIVITY.get(
        node, (node.replace("_", " ").title(), "Workflow", None)
    )
    activity = list(st.session_state.get("activity", []))
    item = next((value for value in activity if value.get("id") == item_id), None)
    started = st.session_state.setdefault("_activity_started", {})
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
    st.session_state.activity = activity
    return item


def _stream_graph_turn(st, graph, config, graph_input, placeholders, vfs_for_ui):
    """Drive one turn of the graph live: `graph_input` is either a fresh state dict (a new chat
    request) or a `Command(resume=...)` carrying the user's chat reply to a thread paused at the
    human-in-the-loop review. Every step redraws the pipeline and activity-log placeholders
    immediately. Returns the clean final state dict — LangGraph's own `"__interrupt__"` marker
    (present only when this turn paused at the review) is stripped out, since its meaning is
    already captured by `stage` staying "Awaiting Approval". It is returned separately, so the
    trace can record where the turn paused."""
    result: dict = {}
    interrupted: tuple = ()
    for chunk in graph.stream(graph_input, config, stream_mode="values"):
        interrupted = chunk.get("__interrupt__") or interrupted
        result = {key: value for key, value in chunk.items() if key != "__interrupt__"}
        live_snapshot = {**result, "activity": list(st.session_state.activity)}
        _render_agent_pipeline(st, placeholders, live_snapshot, vfs_for_ui)
        _render_activity_log(st, placeholders[_ACTIVITY_LOG], st.session_state.activity)
    return result, interrupted


_ACTIVITY_LOG = "__activity_log__"  # placeholders key of the activity log (not an agent row)
_PIPELINE_HEIGHT = 420  # px; the agent rows scroll within this

_PROGRESS_BY_STAGE = {
    "Scoping": "Waiting for your response",
    Stage.AWAITING_APPROVAL.value: "Report ready — reply in the chat to approve or request changes",
    Stage.NEEDS_REVIEW.value: "Needs human review",
    Stage.PUBLISHED.value: "Published",
}


# A published thread is finished. Clarification, approval and recovery states are durable pauses.
_THREAD_DONE_STAGES = {
    Stage.PUBLISHED.value,
    Stage.INCOMPLETE.value,
    "Cancelled",
    "Exported Incomplete",
}


def _chat_turn_trace(text, *, reviewing, clarifying, planning) -> tuple[str, str, dict]:
    """(execution kind, interrupt answered, observation input) of one chat turn."""
    if reviewing:
        return "resume", "publication_approval", {"review_reply": text}
    if planning:
        return "resume", "plan_approval", {"plan_reply": text}
    if clarifying:
        return "resume", "clarification_required", {"clarification_reply": text}
    return "start", "", {"request": text}


def _run_chat_turn(
    st, text, *, reviewing, clarifying=False, planning=False, placeholders, vfs_for_ui
):
    """One chat message drives one graph turn: a new request (or scope answer) starts/continues
    the research; while a report awaits review, the message is the reviewer's reply and resumes
    the paused graph, whose feedback classifier reads it as approve / format / scope.

    The turn is one ``research.execution`` in the run's Langfuse trace (see
    ``observability/tracing.py``); a Streamlit rerun without a message runs no turn."""
    tracer = _get_tracer(st)
    run_id = st.session_state.scope_state["run_id"]
    thread_id = st.session_state.scope_thread_id
    st.session_state._execution_id = str(uuid4())
    resuming = reviewing or clarifying or planning
    kind, interrupt_kind, trace_input = _chat_turn_trace(
        text, reviewing=reviewing, clarifying=clarifying, planning=planning
    )
    with st.status(
        "Processing your reply" if resuming else "Processing research request", expanded=False
    ) as indicator:

        def lifecycle(node, status):
            """Record node progress in the activity log and the status indicator."""
            item = _record_lifecycle(st, node, status)
            indicator.update(label=item["label"], state="running")

        st.session_state._lifecycle_callback = lifecycle
        agent = _get_scope_agent(st)
        if agent is None:
            raise RuntimeError(
                "Reasoning-tier model is not configured. "
                "Set INTELLIGENCE_REASONING_MODEL and retry."
            )
        graph = _get_scope_graph(st, agent, tracer)
        graph_input = (
            Command(resume={"reply": text, "actor": "local-user"})
            if resuming
            else {**st.session_state.scope_state, "request": text, "thread_id": thread_id}
        )
        with tracer.execution(
            run_id=run_id,
            thread_id=thread_id,
            kind=kind,
            interrupt_kind=interrupt_kind,
            input=trace_input,
        ) as execution:
            config = execution.graph_config(
                {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
            )
            try:
                # stream_mode="values" yields the full accumulated state after each node, so
                # every step is redrawn straight into the side panel for live progress. A turn
                # that reaches a fully-passing report pauses at the human-in-the-loop review.
                result, interrupted = _stream_graph_turn(
                    st, graph, config, graph_input, placeholders, vfs_for_ui
                )
            finally:
                st.session_state.pop("_lifecycle_callback", None)
            execution.finish({**result, "__interrupt__": interrupted})
        indicator.update(label="Processing complete", state="complete", expanded=False)
    result["activity"] = list(st.session_state.activity)
    result["trace_id"] = execution.trace_id
    st.session_state.scope_state = result
    st.session_state.stage = result.get("stage", "Scoping")
    st.session_state.progress = _PROGRESS_BY_STAGE.get(
        st.session_state.stage, "Scope clear; Planning started"
    )


def _recovery_service(st):
    tracer = _get_tracer(st)
    agent = _get_scope_agent(st)
    if agent is None:
        raise RuntimeError("Configure the reasoning model before recovering a run.")
    graph = _get_scope_graph(st, agent, tracer)
    return WorkflowRecoveryService(graph, st.session_state.run_registry, tracer)


def _adopt_recovered_state(st, state):
    st.session_state.scope_state = state
    st.session_state.scope_thread_id = state["thread_id"]
    st.session_state.stage = state.get("stage", "Scoping")
    st.session_state.activity = list(state.get("activity", []))
    st.session_state.progress = _PROGRESS_BY_STAGE.get(
        st.session_state.stage, "Recovered checkpoint; continuation available"
    )


def _inspect_checkpoint(st, snapshot):
    if "scope_graph" not in st.session_state or "run_registry" not in st.session_state:
        return None
    service = WorkflowRecoveryService(st.session_state.scope_graph, st.session_state.run_registry)
    try:
        return service.inspect(snapshot.get("thread_id", ""))
    except Exception:
        return None


def _render_checkpoint_tools(st):
    """Find and restore saved runs. Discovery is local and cheap."""
    settings = Settings.from_env()
    if Path(settings.checkpoint_path).is_file():
        if st.button("Find saved checkpoints"):
            try:
                _recovery_service(st).discover_legacy_runs()
                st.rerun()
            except Exception as exc:
                st.error(f"Checkpoint discovery failed: {exc}")
    if Path(settings.run_registry_path).is_file():
        registry = st.session_state.get("run_registry") or RunRegistry(settings.run_registry_path)
        runs = registry.list_unfinished()
        if runs:
            with st.expander("Recover an unfinished run"):
                chosen = st.selectbox(
                    "Run",
                    [run.thread_id for run in runs],
                    format_func=lambda value: next(
                        f"{run.request_summary or run.run_id} — {run.current_stage}"
                        for run in runs
                        if run.thread_id == value
                    ),
                )
                if st.button("Restore checkpoint"):
                    try:
                        _adopt_recovered_state(st, _recovery_service(st).load_state(chosen))
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Recovery failed: {exc}")


def _render_recovery_controls(st, snapshot, checkpoint=None):
    """The run's recovery options, shown right above the chat input and only while the run
    needs review. Widget keys include the draft version, so a new draft starts with an empty
    form; agents are constructed only on an explicit recovery action."""
    if snapshot.get("stage") in _THREAD_DONE_STAGES:
        return
    if _awaiting_recovery(snapshot):
        version = snapshot.get("draft_version", 0)
        # Open only the first time; after a recovery attempt returned a new draft, the draft and
        # the assistant's reply come first and the options stay collapsed until requested.
        tried = any(
            entry.get("kind") == "RECOVERY_DECISION" for entry in snapshot.get("bvf_history") or []
        )
        with st.expander("Recovery options — this run needs your decision", expanded=not tried):
            st.caption("Choose a bounded recovery action. Incomplete export is never publication.")
            action = st.selectbox(
                "Recovery action",
                ["revise", "retry_report", "research", "rescope", "export_incomplete", "cancel"],
                key=f"recovery_action_v{version}",
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
        if apply:
            try:
                if feedback and not validate_request(normalize_text(feedback)).allowed:
                    raise ValueError("Recovery feedback rejected by input guardrails")
                with st.spinner(
                    "Applying the recovery action — rewriting and re-verifying the report"
                ):
                    result = _recovery_service(st).resume_interrupt(
                        snapshot["thread_id"],
                        {"action": action, "feedback": feedback, "additional_attempts": attempts},
                    )
                _adopt_recovered_state(st, result)
                st.rerun()
            except Exception as exc:
                st.error(f"Recovery failed: {exc}")
    elif checkpoint is not None and checkpoint.next_nodes and not checkpoint.has_interrupt:
        if st.button("Continue unfinished run"):
            try:
                _adopt_recovered_state(
                    st, _recovery_service(st).continue_run(snapshot["thread_id"])
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Continuation failed: {exc}")


def render():
    """Render the page and handle the chat input (the Streamlit script entry point)."""
    load_local_env()
    try:
        import streamlit as st
    except ImportError:
        return None
    try:
        st.set_page_config(page_title="Intelligence Collector", layout="wide")
    except Exception:
        pass  # already configured for this script run
    st.title("Intelligence Collector")
    st.session_state.setdefault("stage", "Scoping")
    st.session_state.setdefault("progress", "No research started")
    st.session_state.setdefault("activity", [])
    existing_scope_state = st.session_state.get("scope_state", {})
    st.session_state.setdefault(
        "scope_thread_id", existing_scope_state.get("thread_id", str(uuid4()))
    )
    st.session_state.setdefault(
        "scope_state",
        {
            "run_id": str(uuid4()),
            "thread_id": st.session_state.scope_thread_id,
        },
    )
    snapshot = st.session_state.scope_state
    checkpoint = _inspect_checkpoint(st, snapshot)
    pending = bool(
        checkpoint is not None
        and checkpoint.next_nodes
        and not checkpoint.has_interrupt
        and not checkpoint.terminal
    )
    st.caption(f"{st.session_state.stage} · {st.session_state.progress}")
    tracing_problem = _get_tracer(st).disabled_reason
    if tracing_problem:
        st.caption(f"Langfuse tracing is off ({tracing_problem}); research runs normally.")
    vfs_for_ui = _get_vfs(st)
    chat_col, panel_col = st.columns([3, 2], gap="large")

    # Side panel: the agent pipeline and the activity log, framed apart from the conversation.
    with panel_col, st.container(border=True):
        st.subheader("Pipeline")
        # Fixed height so a long plan or many sources scroll inside the pipeline instead of
        # pushing the activity log and details off screen.
        with st.container(height=_PIPELINE_HEIGHT, border=False):
            placeholders = {agent_key: st.empty() for _, agent_key in _AGENT_ROWS}
            _render_agent_pipeline(st, placeholders, snapshot, vfs_for_ui)
        st.subheader("Activity log")
        placeholders[_ACTIVITY_LOG] = st.empty()
        _render_activity_log(st, placeholders[_ACTIVITY_LOG], list(snapshot.get("activity") or []))
        _render_graph_snapshot(st, snapshot)

    with chat_col:
        _render_checkpoint_tools(st)
        clarification = existing_scope_state.get("scope_assessment", {}).get("question")
        if st.session_state.stage == "Scoping" and clarification:
            st.info("Waiting for your response")
        _render_transcript(st, snapshot)
        planning = _awaiting_plan_approval(checkpoint)
        if planning and snapshot.get("plan"):
            _render_plan(st, snapshot["plan"])
        current_stage = snapshot.get("stage")
        if current_stage == Stage.NEEDS_REVIEW.value:
            st.error(
                "This run needs human review before it can be treated as verified or published."
            )
            st.caption(
                snapshot.get("current_action")
                or snapshot.get("last_error")
                or "Unresolved verification feedback is stored with the run."
            )
        elif current_stage == Stage.PUBLISHED.value:
            publication = snapshot.get("publication") or {}
            st.success(f"Published to {publication.get('destination', '(unknown destination)')}")
        reviewing = _awaiting_review(snapshot)
        clarifying = _awaiting_clarification(snapshot)
        recovering = _awaiting_recovery(snapshot)
        if pending:
            st.info(
                "This checkpoint has unfinished work. Use Continue unfinished run to resume it."
            )
        if current_stage in {"Exported Incomplete", Stage.INCOMPLETE.value}:
            st.warning(
                "Incomplete answer — not a published report. Some requested information could not be verified."
            )
            try:
                st.download_button(
                    "Download incomplete report",
                    vfs_for_ui.read(snapshot["run_id"], "reports/incomplete.md"),
                    file_name="incomplete-report.md",
                    mime="text/markdown",
                )
            except FileNotFoundError:
                st.error("The incomplete report artifact is unavailable.")
        _render_recovery_controls(st, snapshot, checkpoint)
        # Wrapping chat_input in a container (instead of calling it at the script's top level)
        # trades away its viewport-pinned/floating placement so the New Thread button can render
        # in normal flow right below it.
        with st.container():
            text = st.chat_input(
                "Reply: approve, or what to change"
                if reviewing
                else "Reply: approve the plan, cancel, or what to change"
                if planning
                else "Answer the clarification question"
                if clarifying
                else "Use the recovery options above"
                if recovering
                else "What should we research?",
                disabled=recovering or pending or current_stage in _THREAD_DONE_STAGES,
            )
            if st.button("New thread", icon="➕"):
                _start_new_thread(st)
                st.rerun()
        if not text:
            return st
        text = normalize_text(text)
        try:
            decision = validate_request(text)
            if not decision.allowed:
                st.error(decision.reason)
                if not reviewing:
                    st.session_state.progress = "Request rejected; no research started"
                return st
            _run_chat_turn(
                st,
                text,
                reviewing=reviewing,
                clarifying=clarifying,
                planning=planning,
                placeholders=placeholders,
                vfs_for_ui=vfs_for_ui,
            )
            st.rerun()
        except GuardrailViolation as exc:
            if st.session_state.activity:
                st.session_state.activity[-1]["status"] = "failure"
            st.error(f"Request blocked: {exc}")
        except Exception as exc:
            # The turn spans many nodes (Scope through Publish), so name the node that was
            # actually running when it failed instead of always blaming Scope.
            stage_label = _failure_stage_label(st.session_state.activity)
            if st.session_state.activity:
                st.session_state.activity[-1]["status"] = "failure"
            st.error(f"{stage_label} failed: {exc}")
    return st


if __name__ == "__main__":
    render()
