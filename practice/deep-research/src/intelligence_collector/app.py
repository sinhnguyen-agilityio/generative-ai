from __future__ import annotations
import json
import os
import sqlite3
import time
from pathlib import Path
from uuid import uuid4
from intelligence_collector.agents.acceptance_verifier import AcceptanceVerifier
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.config import RoleModelSettings, Settings, load_local_env
from intelligence_collector.contracts import Stage, VerificationResult
from intelligence_collector.guardrails import GuardrailViolation, normalize_text, validate_request
from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.nodes import Node, ResearchNodes
from intelligence_collector.models.factory import ModelFactory
from intelligence_collector.observability.evaluation import make_faithfulness_evaluator
from intelligence_collector.observability.tracing import Tracer
from intelligence_collector.publication import approval_for, can_request_approval
from intelligence_collector.storage.chroma_vector_index import ChromaVectorIndex
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.tools.mcp import TavilyMCP


def _get_tracer(st):
    if "tracer" not in st.session_state:
        st.session_state.tracer = Tracer.from_env(required=True)
    return st.session_state.tracer


def _get_role_router(st, tracer):
    """One shared router, dispatching by role: xhigh reasoning for scope/planner/verify,
    standard for researcher/research_compressor/synthesizer/fixer (see RoleModelSettings)."""
    if "role_router" in st.session_state:
        return st.session_state.role_router
    if not os.getenv("INTELLIGENCE_REASONING_MODEL", "").strip():
        return None
    settings = RoleModelSettings.from_env()
    router = ModelFactory().create_role_router(
        settings, mode="live", trace=tracer.model_event, callbacks=tracer.callbacks,
    )
    st.session_state.role_router = router
    return router


def _get_scope_agent(st, tracer):
    if "scope_agent" in st.session_state:
        return st.session_state.scope_agent
    router = _get_role_router(st, tracer)
    if router is None:
        return None
    agent = ScopeAgent(router)
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
            agent.router,
            # Each research sub-agent gets its own MCP client and per-task search budget.
            mcp_factory=lambda: TavilyMCP(
                tavily_url,
                tavily_key,
                max_calls=settings.max_search_calls,
            ),
            max_search_calls=settings.max_search_calls,
        )
        faithfulness_evaluator = None
        google_api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if google_api_key:
            # RAGAS Faithfulness judge is independent of the app's own reasoning/standard
            # tier models — always Gemini (settings.faithfulness_model, default
            # gemini-3.7-flash), disabled entirely when no Google key is configured.
            faithfulness_evaluator = make_faithfulness_evaluator(
                model=settings.faithfulness_model, api_key=google_api_key,
            )
        research_nodes = ResearchNodes(
            vfs,
            # Real semantic retrieval: Gemini embeddings, persisted locally via Chroma. Used to
            # narrow the Build/Fix prompt to each question's most relevant sources; Verify always
            # re-reads the full evidence independently, so this never affects what gets checked.
            ChromaVectorIndex(
                settings.chroma_path,
                model=settings.embedding_model,
                provider=settings.embedding_provider,
            ),
            researcher,
            # Goal-level check, run only after the deterministic script passes; shares the
            # scope/planner/verify reasoning-tier model (agent.router is the role router).
            acceptance_verifier=AcceptanceVerifier(agent.router),
            max_concurrent_researchers=settings.max_concurrent_researchers,
            recovery_budget=settings.recovery_attempts,
            evidence_top_k=settings.evidence_top_k,
            faithfulness_evaluator=faithfulness_evaluator,
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
    new_thread_id = str(uuid4())
    st.session_state.scope_thread_id = new_thread_id
    st.session_state.scope_state = {"run_id": str(uuid4()), "thread_id": new_thread_id}
    st.session_state.activity = []
    st.session_state.stage = "Scoping"
    st.session_state.progress = "No research started"
    for key in ("verification", "last_trace_id", "publication_decision"):
        st.session_state.pop(key, None)


def _record_value(record, key, default=None):
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _status_text(value):
    return str(getattr(value, "value", value)).replace("_", " ")


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

    activity = list(snapshot.get("activity") or [])
    if activity:
        st.subheader("Activity")
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
            issue = verification.issue_type.value if verification.issue_type else "unresolved"
            st.error(f"Verification: failed — {issue}")
            if verification.repair_details:
                st.write(verification.repair_details)
        if verification.checks:
            st.write("Checks: " + ", ".join(verification.checks))

    faithfulness_score = snapshot.get("faithfulness_score")
    faithfulness_error = snapshot.get("faithfulness_error")
    if faithfulness_score is not None or faithfulness_error:
        st.subheader("Faithfulness (RAGAS)")
        if faithfulness_score is not None:
            st.metric("Faithfulness score", f"{faithfulness_score:.0%}")
            if faithfulness_score < 0.9:
                st.warning("Below the 90% faithfulness target.")
        if faithfulness_error:
            st.caption(f"Faithfulness scoring unavailable: {faithfulness_error}")

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
            st.caption("Checking the report against its sources…")
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


def _render_graph_snapshot(st, snapshot):
    """Keep workflow internals available without competing with the chat."""
    with st.expander("Details", expanded=False):
        verification = _render_details_content(st, snapshot)
        if snapshot.get("trace_id"):
            st.caption(f"Langfuse trace: {snapshot['trace_id']}")
    return verification


def _render_transcript(st, snapshot):
    messages = list(snapshot.get("messages") or [])
    report = snapshot.get("report") or snapshot.get("latest_report") or snapshot.get("draft")
    report_stage = snapshot.get("stage") in {"Awaiting Approval", "Published"}
    last_ai = max(
        (i for i, message in enumerate(messages) if getattr(message, "type", "") == "ai"),
        default=-1,
    )
    for index, message in enumerate(messages):
        kind = getattr(message, "type", "")
        if kind not in {"human", "ai"}:
            continue
        role = "user" if kind == "human" else "assistant"
        content = report if report and report_stage and index == last_ai else message.content
        with st.chat_message(role):
            st.markdown(str(content))


_NODE_ACTIVITY = {
    Node.CLARIFY: ("Evaluating scope", "Scope Agent", None),
    Node.WRITE_BRIEF: ("Generating research brief", "Scope Agent", None),
    Node.PLAN: ("Calling write_todos", "Planner", "write_todos"),
    Node.RESEARCH: ("Delegating research to sub-agents", "Research Supervisor", "tavily_search"),
    Node.ASSESS_RESEARCH: ("Assessing research completeness", "Research Supervisor", "think_tool"),
    Node.SYNTHESIZE: ("Synthesizing report", "Report Writer", None),
    Node.VERIFY: ("Verifying report against sources", "Verifier", "source_verification"),
    Node.EVALUATE_FAITHFULNESS: ("Scoring RAGAS Faithfulness", "Verifier", None),
    Node.FIX: ("Reflecting on verification failures", "Fix Planner", None),
    Node.RESEARCH_GAPS: ("Researching evidence gaps", "Research Supervisor", "tavily_search"),
    Node.NEEDS_REVIEW: ("Escalating for review", "Verifier", None),
    Node.APPROVE: ("Waiting for publication decision", "Publisher", None),
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


def render():
    load_local_env()
    try:
        import streamlit as st
    except ImportError:
        return None
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
    st.caption(f"{st.session_state.stage} · {st.session_state.progress}")
    clarification = existing_scope_state.get("scope_assessment", {}).get(
        "question"
    )
    if st.session_state.stage == "Scoping" and clarification:
        st.info("Waiting for your response")
    _render_transcript(st, st.session_state.scope_state)
    vfs_for_ui = _get_vfs(st)
    st.subheader("Pipeline")
    placeholders = {agent_key: st.empty() for _, agent_key in _AGENT_ROWS}
    _render_agent_pipeline(st, placeholders, st.session_state.scope_state, vfs_for_ui)
    snapshot_verification = _render_graph_snapshot(st, st.session_state.scope_state)
    if st.session_state.scope_state.get("stage") == Stage.NEEDS_REVIEW.value:
        st.error("This run needs human review before it can be treated as verified or published.")
        st.caption(
            st.session_state.scope_state.get("current_action")
            or st.session_state.scope_state.get("last_error")
            or "Unresolved verification feedback is stored with the run."
        )
    verification = snapshot_verification or st.session_state.get("verification")
    if verification is not None and can_request_approval(verification):
        st.subheader("Review and publish")
        decision = st.radio("Publication decision", ["Approve", "Reject"], key="publication_decision")
        if st.button("Record decision"):
            approval = approval_for(
                verification.artifact_hash,
                verification.verification_id,
                decision == "Approve",
                "local-user",
            )
            st.session_state.scope_state = {
                **st.session_state.scope_state,
                "approval": approval.model_dump(mode="json"),
            }
            if approval.approved:
                st.success("Approval recorded; publication remains hash-bound.")
            else:
                st.info("Rejection recorded; permanent output was not changed.")
    # Wrapping chat_input in a container (instead of calling it at the script's top level)
    # trades away its viewport-pinned/floating placement so the New Thread button can render
    # in normal flow right below it.
    with st.container():
        text = st.chat_input("What should we research?")
        if st.button("New thread", icon="➕"):
            _start_new_thread(st)
            st.rerun()
    if text:
        text = normalize_text(text)
        try:
            decision = validate_request(text)
            if not decision.allowed:
                st.error(decision.reason)
                st.session_state.progress = "Request rejected; no research started"
                return st
            tracer = _get_tracer(st)
            run_id = st.session_state.scope_state["run_id"]
            thread_id = st.session_state.scope_thread_id
            execution_id = str(uuid4())
            st.session_state._execution_id = execution_id
            with st.status("Processing research request", expanded=False) as indicator:
                def lifecycle(node, status):
                    item = _record_lifecycle(st, node, status)
                    indicator.update(label=item["label"], state="running")

                st.session_state._lifecycle_callback = lifecycle
                with tracer.execution(
                    run_id,
                    thread_id,
                    input={"request": text},
                ) as execution:
                    agent = _get_scope_agent(st, tracer)
                    if agent is None:
                        raise RuntimeError(
                            "Reasoning-tier model is not configured. "
                            "Set INTELLIGENCE_REASONING_MODEL and retry."
                        )
                    state = {
                        **st.session_state.scope_state,
                        "request": text,
                        "thread_id": thread_id,
                    }
                    config = {"configurable": {"thread_id": thread_id}}
                    graph = _get_scope_graph(st, agent, tracer)
                    result = state
                    try:
                        # stream_mode="values" yields the full accumulated state after each
                        # node finishes (not just its delta), so every step can be redrawn
                        # straight into the pipeline placeholders for live, node-by-node
                        # progress instead of one static view after the whole run completes.
                        for chunk in graph.stream(state, config, stream_mode="values"):
                            result = chunk
                            live_snapshot = {**result, "activity": list(st.session_state.activity)}
                            _render_agent_pipeline(st, placeholders, live_snapshot, vfs_for_ui)
                    finally:
                        st.session_state.pop("_lifecycle_callback", None)
                    execution.output = json.dumps({
                        "stage": result.get("stage"),
                        "need_clarification": result.get("scope_assessment", {}).get(
                            "need_clarification"
                        ),
                        "scope_assessment": result.get("scope_assessment"),
                        "research_brief": result.get("scoped_request"),
                        "messages": [
                            {"role": getattr(message, "type", "message"), "content": getattr(message, "content", str(message))}
                            for message in result.get("messages", [])
                        ],
                    }, ensure_ascii=False, indent=2, default=str)
                indicator.update(label="Processing complete", state="complete", expanded=False)
                result["activity"] = list(st.session_state.activity)
                st.session_state.last_trace_id = execution.trace_id
                result["trace_id"] = execution.trace_id
                st.session_state.scope_state = result
                st.session_state.stage = result.get("stage", "Scoping")
                if st.session_state.stage == "Scoping":
                    st.session_state.progress = "Waiting for your response"
                else:
                    st.session_state.progress = "Scope clear; Planning started"
                st.rerun()
        except GuardrailViolation as exc:
            if st.session_state.activity:
                st.session_state.activity[-1]["status"] = "failure"
            st.error(f"Request blocked: {exc}")
        except Exception as exc:
            if st.session_state.activity:
                st.session_state.activity[-1]["status"] = "failure"
            st.error(f"Scope decision failed: {exc}")
    return st
if __name__ == "__main__": render()
