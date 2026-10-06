"""What the UI derives from a run, tested without Streamlit (ui/run_status.py, ui/evidence.py)."""

from types import SimpleNamespace

import pytest

from intelligence_collector.agents.supervisor import SUPERVISOR_LOG_PATH
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.ui.evidence import bundle_sources, research_counts, todo_bundle
from intelligence_collector.ui.run_status import RunView, agent_status, task_numbers

VERIFIED = {"passed": True, "artifact_hash": "h", "checks": []}


def _checkpoint(**fields):
    return SimpleNamespace(
        **{
            "interrupt_kind": "",
            "next_nodes": [],
            "has_interrupt": False,
            "terminal": False,
            **fields,
        }
    )


@pytest.mark.parametrize(
    ("snapshot", "checkpoint", "flag", "placeholder", "disabled"),
    [
        ({"stage": "Scoping"}, None, None, "What should we research?", False),
        (
            {
                "stage": "Scoping",
                "scope_assessment": {"need_clarification": True, "question": "Q?"},
            },
            None,
            "clarifying",
            "Answer the clarification question",
            False,
        ),
        (
            {"stage": "Planning"},
            _checkpoint(interrupt_kind="plan_approval", has_interrupt=True),
            "planning",
            "What should change in the plan?",
            False,
        ),
        (
            {"stage": "Awaiting Approval", "verification": VERIFIED},
            None,
            "reviewing",
            "What should change in the report?",
            False,
        ),
        ({"stage": "NEEDS_REVIEW"}, None, "recovering", "Use the recovery options above", True),
        (
            {"stage": "Synthesizing"},
            _checkpoint(next_nodes=["synthesizing"]),
            "pending",
            "What should we research?",
            True,
        ),
        ({"stage": "Published"}, None, None, "What should we research?", True),
    ],
)
def test_run_view_knows_who_acts_and_what_the_input_asks(
    snapshot, checkpoint, flag, placeholder, disabled
):
    view = RunView.build(snapshot["stage"], snapshot, checkpoint)
    flags = {"planning", "reviewing", "clarifying", "recovering", "pending"}
    assert {name for name in flags if getattr(view, name)} == ({flag} if flag else set())
    assert view.input_placeholder == placeholder
    assert view.input_disabled is disabled
    assert view.waiting_for_you is (flag in {"planning", "reviewing", "clarifying", "recovering"})


def test_an_unverifiable_report_is_not_offered_for_review():
    view = RunView.build("Awaiting Approval", {"stage": "Awaiting Approval"}, None)
    assert not view.reviewing


def test_agent_status_follows_the_activity_log():
    def snapshot(*entries, **fields):
        return {
            "activity": [{"agent": agent, "status": status} for agent, status in entries],
            **fields,
        }

    assert agent_status(snapshot(), "Planner") == "pending"
    assert agent_status(snapshot(("Planner", "success"), ("Planner", "running")), "Planner") == (
        "running"
    )
    assert agent_status(snapshot(("Planner", "failure")), "Planner") == "failed"
    assert agent_status(snapshot(("Planner", "success")), "Planner") == "complete"
    # A question asked is not a brief written: the Scope Agent is still running.
    assert agent_status(snapshot(("Scope Agent", "success")), "Scope Agent") == "running"
    assert (
        agent_status(
            snapshot(("Scope Agent", "success"), scoped_request={"research_question": "Q"}),
            "Scope Agent",
        )
        == "complete"
    )


def test_task_numbers_follow_the_plan_order():
    assert task_numbers({"todos": [{"todo_id": "b"}, {"todo_id": "a"}]}) == {"b": 1, "a": 2}


def test_evidence_reads_the_latest_round_and_counts_follow_ups(tmp_path):
    vfs = VFS(tmp_path)
    source = {"locator": "https://docs.test/a", "provenance": {"title": "A"}}
    vfs.write_json(
        "run", "research/t1/evidence/result.json", {"sources": [source]}, "evidence_bundle"
    )
    vfs.write_json(
        "run",
        "research/t1/attempts/2/evidence/result.json",
        {
            "sources": [
                source,
                {
                    **source,
                    "locator": "https://docs.test/b",
                    "raw_path": "raw/0.md",
                    "excerpt": 'Summary.\n\nKey excerpts:\n- "Quote."',
                },
                source,  # a repeat is shown once
            ]
        },
        "evidence_bundle",
    )
    vfs.write_json(
        "run",
        SUPERVISOR_LOG_PATH,
        [{"kind": "delegate", "tasks": [{"todo_id": "t1", "follow_up": True}]}],
        "supervisor_log",
    )
    todo = {"todo_id": "t1", "research_attempt": 2}

    sources = bundle_sources(todo_bundle(vfs, "run", todo))

    assert [s.locator for s in sources] == ["https://docs.test/a", "https://docs.test/b"]
    assert sources[1].page_stored and sources[1].summary == "Summary."
    assert todo_bundle(vfs, "run", {"todo_id": "missing"}) == {}
    counts = research_counts(vfs, "run", [todo])
    assert (counts.tasks, counts.sources, counts.follow_ups) == (1, 2, 1)
