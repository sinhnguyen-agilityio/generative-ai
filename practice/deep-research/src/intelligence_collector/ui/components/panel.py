"""The side panel: Run (agent cards, verification gates, quality signals, research limits),
Sources, Activity and Details tabs.

Agent cards and the activity log render into placeholders held by ``LivePanel``, so a running
chat turn redraws them after every graph step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import streamlit as st

from intelligence_collector.agents.supervisor import read_supervisor_log
from intelligence_collector.config import Settings
from intelligence_collector.source_verification import load_catalog
from intelligence_collector.ui.evidence import SourceView, bundle_sources, todo_bundle
from intelligence_collector.ui.html import (
    agent_header_html,
    domain,
    gates_html,
    section_label_html,
    supervisor_log_html,
    verification_gates,
)
from intelligence_collector.ui.run_status import (
    AGENTS,
    agent_status,
    current_report,
    record_value,
    snapshot_verification,
    status_text,
    task_numbers,
)

PIPELINE_HEIGHT = 420  # px; the agent cards scroll within this
QUALITY_METRICS = (
    ("faithfulness", "Faithfulness"),
    ("context_precision", "Context precision"),
    ("response_relevancy", "Response relevancy"),
)


def html(fragment: str) -> None:
    """Render one of ``ui/html.py``'s escaped fragments."""
    st.markdown(fragment, unsafe_allow_html=True)


@dataclass
class LivePanel:
    """Placeholders a running chat turn redraws: one per agent card, plus the activity log."""

    agents: dict
    activity_log: object
    vfs: object

    def refresh(self, snapshot: dict, activity: list) -> None:
        snapshot = {**snapshot, "activity": list(activity)}
        for label, agent in AGENTS:
            _agent_card(self.agents[agent], label, agent, snapshot, self.vfs)
        _activity_log(self.activity_log, activity)


def render_panel(snapshot: dict, vfs) -> LivePanel:
    with st.container(border=True):
        run_tab, sources_tab, activity_tab, details_tab = st.tabs(
            ["Run", "Sources", "Activity", "Details"]
        )
        with run_tab:
            html(section_label_html("Pipeline"))
            # Fixed height: a long plan or many sources scroll here instead of pushing the
            # checks below off screen.
            with st.container(height=PIPELINE_HEIGHT, border=False):
                agents = {agent: st.empty() for _, agent in AGENTS}
            _run_checks(snapshot)
            _research_limits()
        with sources_tab:
            _sources(snapshot, vfs)
        with activity_tab:
            html(section_label_html("Activity log"))
            activity_log = st.empty()
            _supervisor_log(snapshot, vfs)
        with details_tab:
            _details(snapshot)
    panel = LivePanel(agents=agents, activity_log=activity_log, vfs=vfs)
    panel.refresh(snapshot, list(snapshot.get("activity") or []))
    return panel


# ----- Run tab -----


def _agent_card(placeholder, label: str, agent: str, snapshot: dict, vfs) -> None:
    """One card per agent, hidden until that agent starts: the pipeline never lists every
    agent as pending before the user has asked anything."""
    status = agent_status(snapshot, agent)
    if status == "pending":
        placeholder.empty()
        return
    with placeholder.container(border=True):
        html(agent_header_html(label, status))
        if status == "stopped":
            st.caption("Stopped before finishing. Use Continue unfinished run to resume.")
            return
        _AGENT_BODIES[agent](status, snapshot, vfs)


def _scope_body(status: str, snapshot: dict, vfs) -> None:
    if status == "running":
        st.caption("Processing your request…")
    elif status == "complete":
        brief = (snapshot.get("scoped_request") or {}).get("research_question")
        if brief:
            st.write(brief)


def _planner_body(status: str, snapshot: dict, vfs) -> None:
    if status == "running":
        st.caption("Decomposing the brief into research questions…")
    elif status == "complete":
        for todo in snapshot.get("todos") or []:
            st.markdown(f"- {record_value(todo, 'research_question', 'Untitled task')}")


def _research_body(status: str, snapshot: dict, vfs) -> None:
    if status == "running":
        st.caption("Delegating the approved tasks to research sub-agents…")
        return
    if status != "complete":
        return
    run_id = snapshot.get("run_id", "")
    for todo in snapshot.get("todos") or []:
        if status_text(record_value(todo, "status")) != "completed":
            continue
        bundle = todo_bundle(vfs, run_id, todo) if run_id else {}
        sources = bundle_sources(bundle)
        with st.expander(record_value(todo, "research_question", "Untitled task")):
            rounds = int(record_value(todo, "research_attempt", 1) or 1)
            if not bundle:
                verdict = "no evidence bundle"
            elif bundle.get("sufficient"):
                verdict = "sufficient"
            else:
                verdict = "gaps: " + "; ".join(bundle.get("unresolved_gaps") or ["unknown"])
            plural = "s" if len(sources) != 1 else ""
            st.caption(f"Round {rounds} · {len(sources)} source{plural} · {verdict}")
            if not sources:
                st.caption("No sources recorded.")
            for source in sources:
                _source_summary(source)


def _source_summary(source: SourceView) -> None:
    """One source: its link and site, what was stored, and the start of its summary."""
    st.markdown(f"[{source.title}]({source.locator})")
    meta = [domain(source.locator)]
    if source.page_stored:
        meta.append("full page stored and indexed")
    if source.linked:
        meta.append("found by another task")
    st.caption(" · ".join(meta))
    if source.summary:
        st.caption(source.summary[:280] + ("…" if len(source.summary) > 280 else ""))


def _writer_body(status: str, snapshot: dict, vfs) -> None:
    if status == "running":
        st.caption("Drafting the report…")
    elif status == "complete" and snapshot.get("draft"):
        st.caption(f"{len(snapshot['draft'].split())} words drafted — full report in the chat.")


def _verifier_body(status: str, snapshot: dict, vfs) -> None:
    if status == "running":
        st.caption("Judging the report against the evidence…")
        return
    verification = snapshot_verification(snapshot) if status == "complete" else None
    if verification is None:
        return
    if verification.passed:
        st.success("All checks passed")
    else:
        failed = [check.name for check in verification.check_results if not check.passed]
        st.error(f"Failed — {', '.join(failed)}")


_AGENT_BODIES = {
    "Scope Agent": _scope_body,
    "Planner": _planner_body,
    "Research Supervisor": _research_body,
    "Report Writer": _writer_body,
    "Verifier": _verifier_body,
}


def _run_checks(snapshot: dict) -> None:
    """The verification gates of the current draft, then the RAGAS signals."""
    verification = snapshot_verification(snapshot)
    if verification is not None:
        html(section_label_html("Verification gates"))
        html(gates_html(verification_gates(verification, snapshot.get("faithfulness_score"))))
    present = [
        (key, label)
        for key, label in QUALITY_METRICS
        if snapshot.get(f"{key}_score") is not None or snapshot.get(f"{key}_error")
    ]
    if not present:
        return
    html(section_label_html("Report quality (RAGAS) · not a gate"))
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


def _research_limits() -> None:
    settings = Settings.from_env()
    html(section_label_html("Research limits"))
    limits = (
        ("Searches / task", settings.max_search_calls),
        ("Pages / task", settings.max_sources_per_task),
        ("In parallel", settings.max_concurrent_researchers),
    )
    for column, (label, value) in zip(st.columns(len(limits)), limits):
        column.metric(label, value)


# ----- Sources and Activity tabs -----


def _sources(snapshot: dict, vfs) -> None:
    """Every collected source under the [S#] id the report cites, with what the LLMs read."""
    evidence = list(snapshot.get("evidence") or [])
    run_id = snapshot.get("run_id", "")
    if not (evidence and run_id):
        st.caption("Sources appear here as research tasks finish.")
        return
    try:
        catalog = load_catalog(vfs, run_id, evidence)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        st.caption("The stored evidence could not be read.")
        return
    st.caption(f"{len(catalog.sources)} sources collected in this run")
    for sid, source in catalog.sources.items():
        with st.expander(f"{sid} · {source.title or domain(source.url)}"):
            st.markdown(f"[{domain(source.url) or source.url}]({source.url})")
            st.markdown(source.excerpt or "_No excerpt recorded._")


def _activity_log(placeholder, activity: list) -> None:
    """Chronological node log."""
    with placeholder.container():
        if not activity:
            st.caption("No activity yet.")
            return
        for item in activity:
            details = [record_value(item, "agent", "Workflow")]
            if record_value(item, "tool"):
                details.append(f"tool: {record_value(item, 'tool')}")
            if record_value(item, "duration_ms") is not None:
                details.append(f"{record_value(item, 'duration_ms')} ms")
            status = status_text(record_value(item, "status", "unknown"))
            label = record_value(item, "label", "Workflow operation")
            st.markdown(f"- **{status}** — {label} · {' · '.join(details)}")


def _supervisor_log(snapshot: dict, vfs) -> None:
    entries = read_supervisor_log(vfs, snapshot["run_id"]) if snapshot.get("run_id") else []
    if entries:
        html(section_label_html("Research supervisor"))
        html(supervisor_log_html(entries, task_numbers(snapshot)))


# ----- Details tab -----


def _details(snapshot: dict) -> None:
    """Workflow internals, kept apart from the chat; reads the snapshot without changing it."""
    if snapshot.get("current_action"):
        st.write(f"Current action: {snapshot['current_action']}")

    assessment = snapshot.get("progress_assessment")
    if assessment:
        st.subheader("Supervisor assessment")
        next_action = record_value(assessment, "next_action", "")
        blockers = list(record_value(assessment, "blockers", []) or [])
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
                failed = ", ".join(entry.get("failed_checks") or [])
                st.markdown(f"- Cycle {entry.get('attempt')}: {entry['route']} repair of {failed}")

    todos = list(snapshot.get("todos") or [])
    if todos:
        done = sum(status_text(record_value(todo, "status")) == "completed" for todo in todos)
        st.subheader("Research tasks")
        st.write(f"Task progress: {done}/{len(todos)} completed")
        for todo in todos:
            status = status_text(record_value(todo, "status", "unknown"))
            question = record_value(todo, "research_question", "Untitled task")
            st.markdown(f"- **{status}** — {question}")

    verification = snapshot_verification(snapshot)
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

    evidence = list(snapshot.get("evidence") or [])
    if evidence:
        st.subheader("Evidence")
        for item in evidence:
            title = record_value(item, "title") or record_value(item, "source_id", "Evidence")
            locator = record_value(item, "locator") or record_value(item, "path")
            st.markdown(f"- [{title}]({locator})" if locator else f"- {title}")

    report = current_report(snapshot)
    if report:
        st.subheader("Report")
        st.markdown(report)

    if snapshot.get("trace_id"):
        st.caption(f"Langfuse trace: {snapshot['trace_id']}")
