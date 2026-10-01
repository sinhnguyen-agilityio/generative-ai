"""Streamlit UI: the chat on the left and a framed Pipeline / Activity log / Details panel on the
right. Each chat message drives one graph turn; while a report awaits review, the message is the
reviewer's reply.
"""

from __future__ import annotations
import json
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4
from langgraph.types import Command
from intelligence_collector.agents.report_judge import ReportJudge
from intelligence_collector.agents.feedback_router import FeedbackRouter
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.config import RoleModelSettings, Settings, load_local_env
from intelligence_collector.contracts import Stage, VerificationResult
from intelligence_collector.guardrails import GuardrailViolation, normalize_text, validate_request
from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.nodes import Node, ResearchNodes
from intelligence_collector.models.factory import ModelFactory
from intelligence_collector.observability.evaluation import (
    make_context_precision_evaluator,
    make_faithfulness_evaluator,
    make_response_relevancy_evaluator,
)
from intelligence_collector.observability.tracing import Tracer
from intelligence_collector.publication import can_request_approval
from intelligence_collector.storage.chroma_vector_index import ChromaVectorIndex, default_embeddings
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.tools.mcp import TavilyMCP


def _get_tracer(st):
    if "tracer" not in st.session_state:
        st.session_state.tracer = Tracer.from_env(required=True)
    return st.session_state.tracer


def _get_model_routers(st, tracer):
    """Reasoning-tier (xhigh) and standard-tier models, each with its own fallback; every agent
    is handed the tier it needs (see RoleModelSettings)."""
    if "model_routers" in st.session_state:
        return st.session_state.model_routers
    if not os.getenv("INTELLIGENCE_REASONING_MODEL", "").strip():
        return None
    routers = ModelFactory().create_routers(
        RoleModelSettings.from_env(),
        trace=tracer.model_event, callbacks=tracer.callbacks,
    )
    st.session_state.model_routers = routers
    return routers


def _get_scope_agent(st, tracer):
    if "scope_agent" in st.session_state:
        return st.session_state.scope_agent
    routers = _get_model_routers(st, tracer)
    if routers is None:
        return None
    # Scope runs on the reasoning tier; its content-safety guardrail is a cheap binary check.
    agent = ScopeAgent(routers.reasoning, guardrail_router=routers.standard)
    st.session_state.scope_agent = agent
    return agent


def _open_checkpoint_connection(checkpoint_path: str):
    path = Path(checkpoint_path).expanduser()
    if path.exists() and path.is_dir():
        raise RuntimeError(
            f"Checkpoint path points to a directory, not a SQLite file: {path}"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(path, check_same_thread=False)
    except (OSError, sqlite3.Error) as exc:
        raise RuntimeError(
            f"Cannot open SQLite checkpoint database at '{path}'. "
            "Check INTELLIGENCE_CHECKPOINT_PATH and directory write permissions."
        ) from exc


def _quality_evaluators(settings, embeddings) -> dict:
    """RAGAS evaluators for the metrics in ``INTELLIGENCE_EVAL_METRICS``, keyed by their
    ``ResearchNodes`` argument. The judge (default OpenAI gpt-4.1-mini) is independent of the
    app's reasoning/standard tier models; every metric is off when its provider's key is unset.
    Response Relevancy reuses the vector index's embeddings."""
    judge_key_env = "GOOGLE_API_KEY" if settings.faithfulness_provider == "google_genai" else "OPENAI_API_KEY"
    api_key = os.getenv(judge_key_env, "").strip()
    if not api_key:
        return {}
    judge = {"model": settings.faithfulness_model, "api_key": api_key,
             "provider": settings.faithfulness_provider}
    factories = {
        "faithfulness": ("faithfulness_evaluator",
                         lambda: make_faithfulness_evaluator(**judge)),
        "context_precision": ("context_precision_evaluator",
                              lambda: make_context_precision_evaluator(**judge)),
        "response_relevancy": ("response_relevancy_evaluator",
                               lambda: make_response_relevancy_evaluator(**judge, embeddings=embeddings)),
    }
    return {factories[m][0]: factories[m][1]() for m in settings.eval_metrics}


def _get_scope_graph(st, agent, tracer):
    if "scope_graph" in st.session_state:
        return st.session_state.scope_graph
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = Settings.from_env()
    connection = _open_checkpoint_connection(settings.checkpoint_path)
    vfs = VFS(settings.vfs_root)
    research_nodes = None
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    tavily_url = os.getenv("TAVILY_MCP_URL", "").strip()
    if tavily_key and tavily_url:
        researcher = ResearchAgent(
            st.session_state.model_routers.standard,
            # Each research sub-agent gets its own MCP client and per-task search budget.
            mcp_factory=lambda: TavilyMCP(
                tavily_url,
                tavily_key,
                max_calls=settings.max_search_calls,
            ),
            max_search_calls=settings.max_search_calls,
            max_sources=settings.max_sources_per_task,
        )
        embeddings = default_embeddings(settings.embedding_model, settings.embedding_provider)
        research_nodes = ResearchNodes(
            vfs,
            # Real semantic retrieval: Gemini embeddings, persisted locally via Chroma. Used to
            # narrow the Build/Fix prompt to each question's most relevant sources; Verify always
            # re-reads the full evidence independently, so this never affects what gets checked.
            ChromaVectorIndex(settings.chroma_path, embeddings=embeddings),
            researcher,
            # Sole Verify gate (LLM-as-judge, no deterministic pre-check); shares the
            # scope/planner reasoning-tier model.
            report_judge=ReportJudge(agent.router),
            report_guidelines=settings.report_guidelines,
            max_concurrent_researchers=settings.max_concurrent_researchers,
            recovery_budget=settings.recovery_attempts,
            evidence_top_k=settings.evidence_top_k,
            **_quality_evaluators(settings, embeddings),
        )
    graph = build_graph(
        scope_agent=agent,
        vfs=vfs,
        research_nodes=research_nodes,
        checkpointer=SqliteSaver(connection),
        tracer=tracer,
        lifecycle=lambda node, status: st.session_state.get(
            "_lifecycle_callback", lambda *_: None
        )(node, status),
        with_approval=True,
        output_dir=settings.output_dir,
        # A cheap binary classification, so the standard tier (fewer rate-limit failures).
        feedback_router=FeedbackRouter(st.session_state.model_routers.standard),
    )
    st.session_state.scope_checkpoint_connection = connection
    st.session_state.scope_graph = graph
    return graph


def _get_vfs(st):
    if "vfs" not in st.session_state:
        st.session_state.vfs = VFS(Settings.from_env().vfs_root)
    return st.session_state.vfs


def _start_new_thread(st):
    """Reset conversation-scoped state only; cached model/graph/tracer connections stay put."""
    if "tracer" in st.session_state and "scope_thread_id" in st.session_state:
        st.session_state.tracer.end_thread(st.session_state.scope_thread_id)
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
        (key, label) for key, label in _QUALITY_METRICS
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
        st.subheader("Recovery")
        st.write(f"Recovery attempt: {attempt}/{snapshot.get('recovery_budget', 0)}")
        lessons = list(snapshot.get("bvf_lessons") or [])
        if lessons:
            st.write("Lessons learned from failed verifications:")
            for lesson in lessons:
                st.markdown(f"- {lesson}")

    todos = list(snapshot.get("todos") or [])
    if todos:
        completed = sum(
            _status_text(_record_value(todo, "status")) == "completed"
            for todo in todos
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
            title = _record_value(item, "title") or _record_value(
                item, "source_id", "Evidence"
            )
            locator = _record_value(item, "locator") or _record_value(item, "path")
            if locator:
                st.markdown(f"- [{title}]({locator})")
            else:
                st.markdown(f"- {title}")

    report = (
        snapshot.get("report")
        or snapshot.get("latest_report")
        or snapshot.get("draft")
    )
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
                todo for todo in snapshot.get("todos") or []
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


_NODE_ACTIVITY = {
    Node.CLARIFY: ("Evaluating scope", "Scope Agent", None),
    Node.WRITE_BRIEF: ("Generating research brief", "Scope Agent", None),
    Node.PLAN: ("Calling write_todos", "Planner", "write_todos"),
    Node.RESEARCH: ("Delegating research to sub-agents", "Research Supervisor", "tavily_search"),
    Node.ASSESS_RESEARCH: ("Assessing research completeness", "Research Supervisor", "think_tool"),
    Node.SYNTHESIZE: ("Synthesizing report", "Report Writer", None),
    Node.VERIFY: ("Verifying report with LLM judge", "Verifier", "report_judge"),
    Node.EVALUATE_QUALITY: ("Scoring RAGAS report quality", "Verifier", None),
    Node.FIX: ("Reflecting on verification failures", "Fix Planner", None),
    Node.NEEDS_REVIEW: ("Escalating for review", "Verifier", None),
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
    already captured by `stage` staying "Awaiting Approval"."""
    result: dict = {}
    for chunk in graph.stream(graph_input, config, stream_mode="values"):
        result = {key: value for key, value in chunk.items() if key != "__interrupt__"}
        live_snapshot = {**result, "activity": list(st.session_state.activity)}
        _render_agent_pipeline(st, placeholders, live_snapshot, vfs_for_ui)
        _render_activity_log(st, placeholders[_ACTIVITY_LOG], st.session_state.activity)
    return result


_ACTIVITY_LOG = "__activity_log__"  # placeholders key of the activity log (not an agent row)

_PROGRESS_BY_STAGE = {
    "Scoping": "Waiting for your response",
    Stage.AWAITING_APPROVAL.value: "Report ready — reply in the chat to approve or request changes",
    Stage.NEEDS_REVIEW.value: "Needs human review",
    Stage.PUBLISHED.value: "Published",
}


# A thread in one of these stages is finished, so its Langfuse `research.thread` root is ended.
_THREAD_DONE_STAGES = {Stage.PUBLISHED.value, Stage.NEEDS_REVIEW.value}


def _run_chat_turn(st, text, *, reviewing, placeholders, vfs_for_ui):
    """One chat message drives one graph turn: a new request (or scope answer) starts/continues
    the research; while a report awaits review, the message is the reviewer's reply and resumes
    the paused graph, whose feedback classifier reads it as approve / format / scope."""
    tracer = _get_tracer(st)
    run_id = st.session_state.scope_state["run_id"]
    thread_id = st.session_state.scope_thread_id
    st.session_state._execution_id = str(uuid4())
    with st.status("Processing your reply" if reviewing else "Processing research request",
                   expanded=False) as indicator:
        def lifecycle(node, status):
            """Record node progress in the activity log and the status indicator."""
            item = _record_lifecycle(st, node, status)
            indicator.update(label=item["label"], state="running")

        st.session_state._lifecycle_callback = lifecycle
        trace_input = {"review_reply": text} if reviewing else {"request": text}
        # Every turn of one chat thread (request, clarifications, review replies, re-scoping)
        # shares one Langfuse trace, keyed by the thread, and nests under the thread's single
        # `research.thread` root, opened on its first turn and ended once the thread finishes
        # (or on "New thread"), so Langfuse times the whole thread.
        trace_key = f"thread:{thread_id}"
        tracer.start_thread(run_id, thread_id, trace_key=trace_key, input=trace_input)
        with tracer.execution(
            run_id, thread_id, input=trace_input, trace_key=trace_key,
        ) as execution:
            agent = _get_scope_agent(st, tracer)
            if agent is None:
                raise RuntimeError(
                    "Reasoning-tier model is not configured. "
                    "Set INTELLIGENCE_REASONING_MODEL and retry."
                )
            graph_input = (
                Command(resume={"reply": text, "actor": "local-user"}) if reviewing
                else {**st.session_state.scope_state, "request": text, "thread_id": thread_id}
            )
            config = {"configurable": {"thread_id": thread_id}}
            graph = _get_scope_graph(st, agent, tracer)
            try:
                # stream_mode="values" yields the full accumulated state after each node, so
                # every step is redrawn straight into the side panel for live progress. A turn
                # that reaches a fully-passing report pauses at the human-in-the-loop review.
                result = _stream_graph_turn(st, graph, config, graph_input, placeholders, vfs_for_ui)
            finally:
                st.session_state.pop("_lifecycle_callback", None)
            execution.output = json.dumps({
                "stage": result.get("stage"),
                "scope_assessment": result.get("scope_assessment"),
                "research_brief": result.get("scoped_request"),
                "approval": result.get("approval"),
                "feedback_route": result.get("feedback_route"),
                "publication": result.get("publication"),
                "faithfulness_score": result.get("faithfulness_score"),
                "faithfulness_error": result.get("faithfulness_error"),
                "context_precision_score": result.get("context_precision_score"),
                "context_precision_error": result.get("context_precision_error"),
                "response_relevancy_score": result.get("response_relevancy_score"),
                "response_relevancy_error": result.get("response_relevancy_error"),
                "messages": [
                    {"role": getattr(message, "type", "message"),
                     "content": getattr(message, "content", str(message))}
                    for message in result.get("messages", [])
                ],
            }, ensure_ascii=False, indent=2, default=str)
        indicator.update(label="Processing complete", state="complete", expanded=False)
    if result.get("stage") in _THREAD_DONE_STAGES:
        tracer.end_thread(thread_id)
    result["activity"] = list(st.session_state.activity)
    result["trace_id"] = execution.trace_id
    st.session_state.scope_state = result
    st.session_state.stage = result.get("stage", "Scoping")
    st.session_state.progress = _PROGRESS_BY_STAGE.get(
        st.session_state.stage, "Scope clear; Planning started"
    )


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
    st.session_state.setdefault("scope_state", {
        "run_id": str(uuid4()), "thread_id": st.session_state.scope_thread_id,
    })
    snapshot = st.session_state.scope_state
    st.caption(f"{st.session_state.stage} · {st.session_state.progress}")
    vfs_for_ui = _get_vfs(st)
    chat_col, panel_col = st.columns([3, 2], gap="large")

    # Side panel: the agent pipeline and the activity log, framed apart from the conversation.
    with panel_col, st.container(border=True):
        st.subheader("Pipeline")
        placeholders = {agent_key: st.empty() for _, agent_key in _AGENT_ROWS}
        _render_agent_pipeline(st, placeholders, snapshot, vfs_for_ui)
        st.subheader("Activity log")
        placeholders[_ACTIVITY_LOG] = st.empty()
        _render_activity_log(st, placeholders[_ACTIVITY_LOG], list(snapshot.get("activity") or []))
        _render_graph_snapshot(st, snapshot)

    with chat_col:
        clarification = existing_scope_state.get("scope_assessment", {}).get("question")
        if st.session_state.stage == "Scoping" and clarification:
            st.info("Waiting for your response")
        _render_transcript(st, snapshot)
        current_stage = snapshot.get("stage")
        if current_stage == Stage.NEEDS_REVIEW.value:
            st.error("This run needs human review before it can be treated as verified or published.")
            st.caption(
                snapshot.get("current_action")
                or snapshot.get("last_error")
                or "Unresolved verification feedback is stored with the run."
            )
        elif current_stage == Stage.PUBLISHED.value:
            publication = snapshot.get("publication") or {}
            st.success(f"Published to {publication.get('destination', '(unknown destination)')}")
        reviewing = _awaiting_review(snapshot)
        # Wrapping chat_input in a container (instead of calling it at the script's top level)
        # trades away its viewport-pinned/floating placement so the New Thread button can render
        # in normal flow right below it.
        with st.container():
            text = st.chat_input(
                "Reply: approve, or what to change" if reviewing else "What should we research?"
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
                st, text, reviewing=reviewing, placeholders=placeholders, vfs_for_ui=vfs_for_ui,
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
if __name__ == "__main__": render()
