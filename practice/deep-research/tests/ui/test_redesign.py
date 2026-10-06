"""The redesigned UI: header and stepper, plan and report cards with quick replies, the tabbed
side panel (gates, sources, supervisor log)."""

from types import SimpleNamespace

import pytest

from intelligence_collector.agents.supervisor import SUPERVISOR_LOG_PATH
from intelligence_collector.storage.vfs import VFS
from tests.ui.test_chat_progress import _app_with_snapshot, _visible_text

PLAN = {
    "kind": "initial",
    "plan_hash": "9e1c04abcdef",
    "required_answers": ["How LangGraph persists state"],
    "tasks": [
        {
            "todo_id": "t1",
            "question": "How does LangGraph checkpointing work?",
            "evidence_needed": "official persistence documentation",
        }
    ],
    "report_shape": "A short comparison, then sources.",
}
REVIEW = {
    "run_id": "run-ui",
    "thread_id": "thread-ui",
    "stage": "Awaiting Approval",
    "verification": {"passed": True, "artifact_hash": "verified-hash", "checks": []},
    "report": "# Report",
}


@pytest.fixture(autouse=True)
def disable_local_dotenv(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")


@pytest.fixture
def chat_turns(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "intelligence_collector.ui.turn.run_chat_turn",
        lambda reply, **kwargs: calls.append(
            (reply, {key: getattr(kwargs["view"], key) for key in ("reviewing", "planning")})
        ),
    )
    return calls


@pytest.fixture
def plan_interrupt(monkeypatch):
    checkpoint = SimpleNamespace(
        has_interrupt=True,
        interrupt_kind="plan_approval",
        next_nodes=["await_plan_approval"],
        terminal=False,
    )
    monkeypatch.setattr(
        "intelligence_collector.ui.resources.inspect_checkpoint", lambda snapshot: checkpoint
    )


def _plan_app():
    return _app_with_snapshot(
        {"run_id": "run-ui", "thread_id": "thread-ui", "stage": "Planning", "plan": PLAN}
    )


def test_header_shows_the_stage_and_who_has_to_act(plan_interrupt):
    text = _visible_text(_plan_app())
    assert "Intelligence Collector" in text
    assert '<span class="ic-chip ic-wait">Planning</span>' in text  # waiting for you
    assert '<li class="ic-step wait"><span class="ic-dot">2</span>Plan</li>' in text


def test_plan_card_lists_tasks_and_approves_with_one_click(plan_interrupt, chat_turns):
    app = _plan_app()
    card = app.chat_message[-1]
    card_text = "\n".join(str(m.value) for m in list(card.markdown) + list(card.caption))
    assert "**Research plan** · hash 9e1c04" in card_text
    assert "**1.** How does LangGraph checkpointing work?" in card_text
    assert "Evidence: official persistence documentation" in card_text
    assert app.chat_input[0].placeholder == "What should change in the plan?"

    next(b for b in app.button if b.label == "Approve plan").click().run()

    assert not app.exception
    assert chat_turns == [(("approve", True), {"reviewing": False, "planning": True})]


def test_plan_card_cancel_is_explicit_and_request_changes_is_a_chat_reply(
    plan_interrupt, chat_turns
):
    app = _plan_app()
    next(b for b in app.button if b.label == "Cancel run").click().run()
    app.text_area(key="plan_change").input("Add a task on Temporal pricing.").run()
    next(b for b in app.button if b.label == "Send").click().run()

    assert [reply for reply, _ in chat_turns] == [
        ("cancel", False),
        ("Add a task on Temporal pricing.", None),
    ]


def test_report_card_offers_approve_download_and_changes(chat_turns):
    app = _app_with_snapshot(REVIEW)
    labels = [b.label for b in app.button]
    assert "Approve and publish" in labels
    assert app.get("download_button")[0].proto.label == "Download .md"

    next(b for b in app.button if b.label == "Approve and publish").click().run()

    assert chat_turns == [(("approve", True), {"reviewing": True, "planning": False})]


def test_run_tab_shows_the_three_verification_gates():
    app = _app_with_snapshot(
        {
            **REVIEW,
            "faithfulness_score": 0.96,
            "verification": {
                "passed": True,
                "artifact_hash": "h",
                "check_results": [
                    {"name": "c1", "passed": True, "gate": "coverage", "item": "a"},
                    {"name": "cit", "passed": True, "gate": "citation"},
                    {"name": "f", "passed": True, "gate": "faithfulness"},
                ],
            },
        }
    )
    run_text = "\n".join(str(m.value) for m in app.tabs[0].markdown)
    assert "Verification gates" in run_text
    assert "1 of 1 required items answered" in run_text
    assert '<span class="ic-mono">0.96</span>' in run_text
    assert 'class="ic-gate fail"' not in run_text


def _store_research(tmp_path):
    vfs = VFS(str(tmp_path))
    ref = vfs.write_json(
        "run-ui",
        "research/t1/evidence/result.json",
        {
            "task_id": "t1",
            "question": "How does LangGraph checkpointing work?",
            "summary": "Checkpoints per step [https://docs.test/persistence]",
            "sufficient": True,
            "unresolved_gaps": [],
            "sources": [
                {
                    "locator": "https://docs.test/persistence",
                    "provenance": {"title": "Persistence"},
                    "excerpt": 'Checkpoints save state.\n\nKey excerpts:\n- "A checkpoint saves state."',
                    "raw_path": "research/t1/raw/0.md",
                }
            ],
        },
        "evidence_bundle",
        task_id="t1",
    )
    vfs.write_json(
        "run-ui",
        SUPERVISOR_LOG_PATH,
        [
            {"kind": "think", "text": "One task; research it."},
            {"kind": "delegate", "tasks": [{"todo_id": "t1", "round": 1, "follow_up": False}]},
            {
                "kind": "result",
                "todo_id": "t1",
                "status": "completed",
                "sufficient": True,
                "sources": 1,
            },
            {
                "kind": "delegate",
                "tasks": [
                    {"todo_id": "t1", "round": 2, "follow_up": True, "topic": "Find limits."}
                ],
            },
            {"kind": "complete"},
        ],
        "supervisor_log",
    )
    return ref


def test_sources_tab_banner_and_supervisor_log_come_from_the_stored_research(tmp_path, monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_VFS_ROOT", str(tmp_path))
    ref = _store_research(tmp_path)
    app = _app_with_snapshot(
        {
            **REVIEW,
            "todos": [
                {
                    "todo_id": "t1",
                    "research_question": "How does LangGraph checkpointing work?",
                    "status": "completed",
                }
            ],
            "evidence": [ref.model_dump(mode="json")],
            "activity": [
                {
                    "id": "research",
                    "label": "Delegating research to sub-agents",
                    "agent": "Research Supervisor",
                    "status": "success",
                }
            ],
        }
    )

    assert not app.exception
    sources = app.tabs[1]
    assert sources.expander[0].label == "S1 · Persistence"
    assert "A checkpoint saves state." in "\n".join(m.value for m in sources.markdown)
    text = _visible_text(app)
    assert "<b>Research finished</b> · 1 task · 1 source · 1 follow-up" in text
    activity = "\n".join(m.value for m in app.tabs[2].markdown)
    assert "Research supervisor" in activity
    assert "One task; research it." in activity and "Research complete" in activity
    # The research card in the Run tab marks the page as stored.
    assert "full page stored and indexed" in "\n".join(c.value for c in app.caption)
