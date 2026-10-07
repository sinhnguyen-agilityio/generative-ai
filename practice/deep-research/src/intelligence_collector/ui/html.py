"""HTML fragments of the Streamlit UI (header, stage stepper, status chips, verification gates,
supervisor log) and the theme's CSS from ``theme.css``.

Every function here is pure — snapshot in, HTML string out — so the look is testable without
Streamlit. All dynamic text is escaped: research questions, page titles and model reflections
come from users and web pages.
"""

from __future__ import annotations

from functools import lru_cache
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

from intelligence_collector.contracts import Stage

THEME_CSS_PATH = Path(__file__).with_name("theme.css")


@lru_cache(maxsize=1)
def theme_css() -> str:
    """The theme's CSS as a ``<style>`` block (read once per process)."""
    return f"<style>\n{THEME_CSS_PATH.read_text(encoding='utf-8')}</style>"


STEPS = ("Scope", "Plan", "Research", "Write", "Verify", "Publish")

# Where each stage sits on the stepper.
_STAGE_STEP = {
    Stage.SCOPING.value: 0,
    Stage.PLANNING.value: 1,
    Stage.RESEARCHING.value: 2,
    Stage.SYNTHESIZING.value: 3,
    Stage.VERIFYING.value: 4,
    Stage.NEEDS_REVIEW.value: 4,
    Stage.FAILED.value: 4,
    Stage.AWAITING_APPROVAL.value: 5,
    Stage.PUBLISHED.value: 5,
    Stage.INCOMPLETE.value: 5,
    Stage.EXPORTED_INCOMPLETE.value: 5,
    Stage.CANCELLED.value: 1,
}

_STAGE_LABEL = {
    Stage.NEEDS_REVIEW.value: "Needs review",
    Stage.AWAITING_APPROVAL.value: "Verified · awaiting your approval",
}


def stage_view(
    stage: str, *, has_request: bool, waiting_for_you: bool
) -> tuple[str, str, list[tuple[str, str]]]:
    """``(label, tone, steps)`` for the header chip and the stepper. Tones: ``ok`` (done),
    ``run`` (the system is working), ``wait`` (waiting for the user), ``fail``, ``idle``."""
    current = _STAGE_STEP.get(stage, 0)
    if stage == Stage.PUBLISHED.value:
        tone, current = "ok", len(STEPS)
    elif stage in {
        Stage.NEEDS_REVIEW.value,
        Stage.FAILED.value,
        Stage.INCOMPLETE.value,
        Stage.EXPORTED_INCOMPLETE.value,
        Stage.CANCELLED.value,
    }:
        tone = "fail"
    elif waiting_for_you:
        tone = "wait"
    elif stage == Stage.SCOPING.value and not has_request:
        tone = "idle"
    else:
        tone = "run"
    step_state = {"ok": "done", "run": "run", "wait": "wait", "fail": "fail", "idle": "todo"}
    steps = [
        (name, "done" if index < current else step_state[tone] if index == current else "todo")
        for index, name in enumerate(STEPS)
    ]
    return _STAGE_LABEL.get(stage, stage), tone, steps


def chip_html(text: str, tone: str) -> str:
    return f'<span class="ic-chip ic-{tone}">{escape(text)}</span>'


def header_html(label: str, tone: str, progress: str, thread_id: str, run_id: str) -> str:
    """App name, the thread and run ids, the stage chip and the progress line."""
    return (
        '<div class="ic-head">'
        '<svg width="26" height="26" viewBox="0 0 28 28" fill="none" stroke="#1F4FB8" '
        'stroke-width="2" aria-hidden="true"><rect x="3" y="3" width="22" height="22" rx="6">'
        '</rect><circle cx="12.5" cy="12.5" r="4.5"></circle><path d="M16 16l5 5"></path></svg>'
        '<span class="ic-brand">Intelligence Collector</span>'
        f'<span class="ic-ids">thread {escape(thread_id[:8])} · run {escape(run_id[:8])}</span>'
        f"{chip_html(label, tone)}"
        f'<span class="ic-progress">{escape(progress)}</span>'
        "</div>"
    )


def stepper_html(steps: list[tuple[str, str]]) -> str:
    """The six run stages as numbered steps joined by bars; finished steps show a check."""
    items = []
    for index, (name, state) in enumerate(steps):
        if index:
            bar = "done" if steps[index - 1][1] == "done" else ""
            items.append(f'<li aria-hidden="true" class="ic-bar {bar}"></li>')
        mark = "&#10003;" if state == "done" else str(index + 1)
        items.append(
            f'<li class="ic-step {state}"><span class="ic-dot">{mark}</span>{escape(name)}</li>'
        )
    return f'<ol class="ic-steps" aria-label="Run stages">{"".join(items)}</ol>'


AGENT_STATUS = {
    "pending": ("Pending", "idle"),
    "running": ("Running", "run"),
    "complete": ("Complete", "ok"),
    "failed": ("Failed", "fail"),
    "stopped": ("Stopped", "fail"),
}


def agent_header_html(label: str, status: str) -> str:
    text, tone = AGENT_STATUS[status]
    return (
        f'<div class="ic-agent"><span class="ic-agent-name">{escape(label)}</span>'
        f"{chip_html(text, tone)}</div>"
    )


def section_label_html(text: str) -> str:
    return f'<div class="ic-label">{escape(text)}</div>'


def verification_gates(verification, claim_support_score) -> list[dict]:
    """The three verification gates of one verified draft: coverage, citations, faithfulness.

    Each gate: ``name``, ``passed`` (None when that gate has no recorded result) and
    ``detail``; faithfulness also carries its ``score``: the share of factual sentences the
    claim judge found supported by their own cited sources (spec 002, FR-026). The gate passes
    only at full support; RAGAS faithfulness is a quality signal shown elsewhere."""
    results = list(getattr(verification, "check_results", None) or [])
    by_gate: dict[str, list] = {}
    for check in results:
        by_gate.setdefault(check.gate or "", []).append(check)

    def gate_passed(name: str):
        checks = by_gate.get(name)
        if checks:
            return all(check.passed for check in checks)
        return bool(verification.passed) if verification is not None and not results else None

    coverage = by_gate.get("coverage", [])
    answered = sum(check.passed for check in coverage)
    gates = [
        {
            "name": "Coverage",
            "passed": gate_passed("coverage"),
            "detail": f"{answered} of {len(coverage)} required items answered"
            if coverage
            else "Every required answer is covered by the report",
        },
        {
            "name": "Citations",
            "passed": gate_passed("citation"),
            "detail": "Every [S#] resolves to a collected source",
        },
    ]
    score = claim_support_score
    passed = gate_passed("faithfulness")
    if score is not None and passed is None:
        passed = score >= 1.0
    failing = sum(len(check.details) for check in by_gate.get("faithfulness", []))
    gates.append(
        {
            "name": "Faithfulness",
            "passed": passed,
            "score": score,
            "detail": f"{failing} sentence{'s' if failing != 1 else ''} not supported by "
            "the sources it cites"
            if failing
            else "Every factual sentence is supported by the sources it cites",
        }
    )
    return gates


def gates_html(gates: list[dict]) -> str:
    parts = []
    for gate in gates:
        failed = gate["passed"] is False
        icon = (
            '<svg width="20" height="20" viewBox="0 0 20 20" fill="none" '
            f'stroke="{"#B42318" if failed else "#1F7A4D"}" stroke-width="2" aria-hidden="true">'
            '<circle cx="10" cy="10" r="8"></circle>'
            + (
                '<path d="M7 7l6 6M13 7l-6 6"></path>'
                if failed
                else '<path d="M6.5 10.5l2.5 2.5 4.5-5"></path>'
            )
            + "</svg>"
        )
        status = "Not run" if gate["passed"] is None else "Failed" if failed else "Passed"
        score = gate.get("score")
        value = f'<span class="ic-mono">{score:.2f}</span>' if score is not None else status
        meter = (
            f'<div class="ic-meter" aria-hidden="true"><span style="width:{max(0.0, min(score, 1.0)) * 100:.0f}%">'
            "</span></div>"
            if score is not None
            else ""
        )
        parts.append(
            f'<div class="ic-gate{" fail" if failed else ""}">{icon}<div class="ic-gate-body">'
            f'<div class="ic-gate-name"><span>{escape(gate["name"])}</span>{value}</div>'
            f'{meter}<div class="ic-gate-detail">{escape(gate["detail"])}</div></div></div>'
        )
    return "".join(parts)


def supervisor_log_html(entries: list[dict], task_numbers: dict[str, int] | None = None) -> str:
    """The supervisor's decisions as a timeline: reflections, delegations, task results,
    refusals and the final decision."""
    numbers = task_numbers or {}

    def task_name(todo_id: str) -> str:
        return f"Task {numbers[todo_id]}" if todo_id in numbers else f"Task {todo_id[:8]}"

    rows = []
    for entry in entries:
        kind = entry.get("kind")
        if kind == "think":
            tool, css = "think_tool", "ic-think"
            body = f'<span class="ic-quote">“{escape(str(entry.get("text", "")))}”</span>'
        elif kind == "delegate":
            tasks = entry.get("tasks") or []
            tool, css = (
                ("auto-delegate" if entry.get("automatic") else f"Conduct ×{len(tasks)}"),
                "ic-act",
            )
            names = ", ".join(
                task_name(task["todo_id"])
                + (f" (round {task['round']})" if task.get("follow_up") else "")
                for task in tasks
            )
            topic = next((task.get("topic") for task in tasks if task.get("follow_up")), "")
            body = f"Delegated {escape(names)}"
            if topic:
                body += f'<div class="ic-muted">Missing evidence: {escape(topic)}</div>'
        elif kind == "result":
            tool, css = "result", "ic-note"
            if entry.get("error") is not None or entry.get("status") == "failed":
                tone, text = "fail", "no usable evidence"
            elif entry.get("sufficient"):
                tone, text = "ok", f"sufficient · {entry.get('sources', 0)} pages"
            else:
                tone, text = "wait", f"gap found · {entry.get('sources', 0)} pages"
            gaps = entry.get("gaps") or []
            body = chip_html(f"{task_name(entry.get('todo_id', ''))} · {text}", tone)
            if gaps:
                body += f'<div class="ic-muted">Gaps: {escape("; ".join(gaps))}</div>'
        elif kind == "complete":
            tool, css, body = "Complete", "ic-act", "Research complete"
        elif kind == "replan":
            tool, css = "ProposeReplan", "ic-act"
            body = f"Replacement plan proposed: {escape(str(entry.get('text', '')))}"
        elif kind == "refused":
            tool, css = "refused", "ic-note"
            body = f'<span class="ic-muted">{escape(str(entry.get("text", "")))}</span>'
        else:
            continue
        rows.append(f'<li><span class="ic-tool {css}">{escape(tool)}</span><div>{body}</div></li>')
    return f'<ol class="ic-log">{"".join(rows)}</ol>' if rows else ""


def research_banner_html(tasks: int, sources: int, follow_ups: int) -> str:
    """One line in the chat once research has finished."""
    extra = f" · {follow_ups} follow-up{'s' if follow_ups != 1 else ''}" if follow_ups else ""
    return (
        '<div class="ic-banner"><svg width="16" height="16" viewBox="0 0 16 16" fill="none" '
        'stroke="#1F7A4D" stroke-width="2" aria-hidden="true"><path d="M3.5 8.5l3 3L12.5 5">'
        f"</path></svg><span><b>Research finished</b> · {tasks} task{'s' if tasks != 1 else ''}"
        f" · {sources} source{'s' if sources != 1 else ''}{extra}</span></div>"
    )


def domain(url: str) -> str:
    return urlsplit(url).netloc.removeprefix("www.")


def _label(text: str) -> str:
    return text.replace("_", " ")


def task_research_html(research) -> str:
    """One task's searches (``ui.evidence.TaskResearch``): queries, accepted and rejected
    results with reasons, why it stopped, and its report's status and open points."""
    rejected = sum(research.rejected.values())
    reasons = ", ".join(
        f"{_label(reason)} {count}" for reason, count in sorted(research.rejected.items())
    )
    lines = [
        f"{research.accepted} accepted · {rejected} rejected" + (f" ({escape(reasons)})" if reasons else "")
    ]
    status = []
    if research.stop_reason:
        status.append(f"stopped: {escape(_label(research.stop_reason))}")
    if research.report_status:
        status.append(f"report: {escape(research.report_status)}")
    if status:
        lines.append(" · ".join(status))
    if research.open_points:
        lines.append("Open: " + escape(", ".join(research.open_points)))
    queries = "".join(f"<li>{escape(query)}</li>" for query in research.queries)
    body = "".join(f'<div class="ic-muted">{line}</div>' for line in lines)
    if queries:
        body += f'<div class="ic-muted">Queries:</div><ol class="ic-muted">{queries}</ol>'
    return body


def rounds_html(rounds) -> str:
    """The research rounds (``ui.evidence.RoundView``) as a timeline: how each round's gap
    analysis ended, the essential gaps still open and the follow-up tasks it created."""
    rows = []
    for view in rounds:
        body = escape(f"Round {view.round}: {_label(view.ended) or 'analyzed'}")
        if view.closed_essential:
            body += escape(f" · {view.closed_essential} essential gap(s) closed")
        for follow_up in view.follow_ups:
            gaps = ", ".join(follow_up.gap_point_ids)
            body += (
                f'<div class="ic-muted">Follow-up: {escape(follow_up.question)}'
                + (f" · gaps {escape(gaps)}" if gaps else "")
                + "</div>"
            )
        for gap in view.open_essential:
            body += f'<div class="ic-muted">Open: {escape(gap)}</div>'
        for refusal in view.refusals:
            body += f'<div class="ic-muted">Refused: {escape(refusal)}</div>'
        rows.append(
            f'<li><span class="ic-tool ic-act">gap_analysis</span><div>{body}</div></li>'
        )
    return f'<ol class="ic-log">{"".join(rows)}</ol>' if rows else ""


def claim_html(claim) -> str:
    """One judged sentence (``ui.evidence.ClaimView``): its verdict and why it failed."""
    tone = "ok" if claim.passed else "fail"
    verdict = _label(claim.verdict) or "judged"
    body = chip_html(f"{claim.claim_id} · {verdict}", tone)
    body += f'<div class="ic-quote">{escape(claim.sentence)}</div>'
    if claim.failures:
        body += f'<div class="ic-muted">Failed: {escape("; ".join(claim.failures))}</div>'
    return body


def cited_text_html(source) -> str:
    """The header of one cited source under a claim; the text itself is shown as plain text."""
    label = ' <span class="ic-cite">snippet only</span>' if source.snippet_only else ""
    link = (
        f'<a href="{escape(source.url, quote=True)}">{escape(source.title)}</a>'
        if source.url
        else escape(source.title)
    )
    return f'<div><span class="ic-cite">{escape(source.source_id)}</span> {link}{label}</div>'
