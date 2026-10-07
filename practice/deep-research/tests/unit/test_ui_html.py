"""UI presentation helpers: stage tones, stepper, verification gates and the supervisor log."""

from intelligence_collector.contracts import VerificationCheck, VerificationResult
from intelligence_collector.ui.html import (
    gates_html,
    header_html,
    research_banner_html,
    stage_view,
    stepper_html,
    supervisor_log_html,
    verification_gates,
)


def test_stage_view_colours_who_has_to_act():
    label, tone, steps = stage_view("Scoping", has_request=False, waiting_for_you=False)
    assert (label, tone) == ("Scoping", "idle")
    assert [state for _, state in steps] == ["todo"] * 6

    _, tone, steps = stage_view("Planning", has_request=True, waiting_for_you=True)
    assert tone == "wait"  # the plan waits for the user
    assert [state for _, state in steps] == ["done", "wait", "todo", "todo", "todo", "todo"]

    _, tone, steps = stage_view("Researching", has_request=True, waiting_for_you=False)
    assert tone == "run"
    assert [state for _, state in steps][:3] == ["done", "done", "run"]

    label, tone, steps = stage_view("Awaiting Approval", has_request=True, waiting_for_you=True)
    assert (label, tone) == ("Verified · awaiting your approval", "wait")
    assert [state for _, state in steps] == ["done"] * 5 + ["wait"]

    label, tone, steps = stage_view("NEEDS_REVIEW", has_request=True, waiting_for_you=True)
    assert (label, tone) == ("Needs review", "fail")
    assert steps[4] == ("Verify", "fail")

    _, tone, steps = stage_view("Published", has_request=True, waiting_for_you=False)
    assert tone == "ok" and all(state == "done" for _, state in steps)


def test_stepper_marks_finished_steps_and_joins_them():
    html = stepper_html(stage_view("Researching", has_request=True, waiting_for_you=False)[2])
    assert html.count('class="ic-step done"') == 2
    assert html.count('class="ic-bar done"') == 2  # Scope-Plan and Plan-Research
    assert '<li class="ic-step run"><span class="ic-dot">3</span>Research</li>' in html


def test_header_escapes_untrusted_text():
    html = header_html("Planning", "wait", "<script>alert(1)</script>", "t<b>", "run")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "thread t&lt;b&gt;" in html


def _check(gate, passed, item=""):
    return VerificationCheck(name=f"{gate}:{item}", passed=passed, gate=gate, item=item)


def test_gates_summarize_coverage_citations_and_faithfulness():
    verification = VerificationResult(
        artifact_hash="h",
        passed=False,
        check_results=[
            _check("coverage", True, "a"),
            _check("coverage", False, "b"),
            _check("citation", True),
            _check("faithfulness", True),
        ],
    )

    coverage, citations, faithfulness = verification_gates(verification, 1.0)

    assert coverage == {
        "name": "Coverage",
        "passed": False,
        "detail": "1 of 2 required items answered",
    }
    assert citations["passed"] is True
    assert faithfulness["passed"] is True and faithfulness["score"] == 1.0
    assert faithfulness["detail"] == "Every factual sentence is supported by the sources it cites"
    html = gates_html([coverage, citations, faithfulness])
    assert html.count('class="ic-gate fail"') == 1
    assert 'style="width:100%"' in html


def test_faithfulness_gate_counts_the_failing_claims():
    verification = VerificationResult(
        artifact_hash="h",
        passed=False,
        check_results=[
            VerificationCheck(
                name="claims", passed=False, gate="faithfulness",
                details=["K1: uncited — x", "K3: contradicted — y"],
            )
        ],
    )
    *_, faithfulness = verification_gates(verification, 0.6)
    assert faithfulness["passed"] is False and faithfulness["score"] == 0.6
    assert faithfulness["detail"] == "2 sentences not supported by the sources it cites"


def test_claim_support_alone_passes_only_at_full_support():
    verification = VerificationResult(
        artifact_hash="h", passed=False, check_results=[_check("coverage", True, "a")]
    )
    assert verification_gates(verification, 0.95)[-1]["passed"] is False
    assert verification_gates(verification, 1.0)[-1]["passed"] is True


def test_supervisor_log_shows_decisions_in_order_and_escapes_reflections():
    entries = [
        {"kind": "think", "text": "Research both <now>."},
        {
            "kind": "delegate",
            "automatic": False,
            "tasks": [
                {"todo_id": "a", "round": 1, "follow_up": False, "topic": "x"},
                {"todo_id": "b", "round": 1, "follow_up": False, "topic": "y"},
            ],
        },
        {"kind": "result", "todo_id": "a", "status": "completed", "sufficient": True, "sources": 5},
        {
            "kind": "result",
            "todo_id": "b",
            "status": "completed",
            "sufficient": False,
            "sources": 3,
            "gaps": ["release date"],
        },
        {
            "kind": "delegate",
            "tasks": [{"todo_id": "b", "round": 2, "follow_up": True, "topic": "Find the date."}],
        },
        {"kind": "result", "todo_id": "b", "status": "completed", "error": "timeout"},
        {"kind": "complete"},
    ]

    html = supervisor_log_html(entries, {"a": 1, "b": 2})

    assert "&lt;now&gt;" in html and "<now>" not in html
    order = [
        "think_tool",
        "Conduct ×2",
        "Task 1 · sufficient · 5 pages",
        "Task 2 · gap found · 3 pages",
        "Gaps: release date",
        "Delegated Task 2 (round 2)",
        "Missing evidence: Find the date.",
        "Task 2 · no usable evidence",
        "Research complete",
    ]
    positions = [html.index(text) for text in order]
    assert positions == sorted(positions)


def test_research_banner_counts_tasks_sources_and_follow_ups():
    assert "3 tasks · 14 sources · 1 follow-up" in research_banner_html(3, 14, 1)
    assert "follow-up" not in research_banner_html(1, 1, 0)
