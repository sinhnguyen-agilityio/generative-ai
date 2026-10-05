from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from intelligence_collector.storage.vfs import VFS
from tests.ui.test_chat_progress import _app_with_snapshot, _visible_text

APP_PATH = Path(__file__).parents[2] / "app.py"


@pytest.fixture(autouse=True)
def disable_local_dotenv(monkeypatch):
    """UI contract tests must never load a developer's real local credentials."""
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")


def test_pipeline_rows_stay_hidden_until_their_agent_starts_processing():
    """No agent has produced any activity yet, so the pipeline section header is shown but
    no per-agent row (and no "Pending" badge) appears until that agent actually starts."""
    app = AppTest.from_file(APP_PATH).run()

    text = _visible_text(app)
    assert "Pipeline" in text
    assert "Scope Agent" not in text
    assert "Researcher" not in text
    assert "Pending" not in text


def test_scope_agent_row_appears_once_it_starts_processing():
    app = _app_with_snapshot(
        {
            "run_id": "run-ui",
            "thread_id": "thread-ui",
            "stage": "Scoping",
            "activity": [
                {
                    "id": "clarify",
                    "label": "Evaluating scope",
                    "agent": "Scope Agent",
                    "status": "running",
                },
            ],
        }
    )

    text = _visible_text(app)
    assert 'Scope Agent</span><span class="ic-chip ic-run">Running</span>' in text
    assert "Planner" not in text


def test_new_thread_button_starts_a_fresh_run_and_thread():
    app = AppTest.from_file(APP_PATH).run()
    first_thread_id = app.session_state["scope_thread_id"]
    first_run_id = app.session_state["scope_state"]["run_id"]
    app.session_state["activity"] = [
        {"id": "x", "label": "Evaluating scope", "agent": "Scope Agent", "status": "success"}
    ]
    app.session_state["stage"] = "Planning"

    new_thread_button = next(b for b in app.button if "New thread" in b.label)
    new_thread_button.click().run()

    assert app.session_state["scope_thread_id"] != first_thread_id
    assert app.session_state["scope_state"]["run_id"] != first_run_id
    assert app.session_state["scope_state"]["thread_id"] == app.session_state["scope_thread_id"]
    assert app.session_state["activity"] == []
    assert app.session_state["stage"] == "Scoping"
    assert app.session_state["progress"] == "No research started"


def test_scope_agent_brief_appears_only_once_a_research_brief_exists():
    app = _app_with_snapshot(
        {
            "run_id": "run-ui",
            "thread_id": "thread-ui",
            "stage": "Planning",
            "scoped_request": {"research_question": "I want to compare X and Y."},
            "activity": [
                {
                    "id": "clarify",
                    "label": "Evaluating scope",
                    "agent": "Scope Agent",
                    "status": "success",
                },
                {
                    "id": "plan",
                    "label": "Calling write_todos",
                    "agent": "Planner",
                    "status": "running",
                },
            ],
        }
    )

    text = _visible_text(app)
    # One card per agent: its name with its status chip, then its content — before the next
    # agent's card (Planner) begins.
    assert (
        text.index('Scope Agent</span><span class="ic-chip ic-ok">Complete</span>')
        < text.index("I want to compare X and Y.")
        < text.index("Planner")
    )


def test_scope_agent_is_still_running_while_only_a_clarifying_question_was_asked():
    """A successful clarify_with_user turn that only asked a question is not 'complete' —
    the Scope Agent hasn't produced a brief yet."""
    app = _app_with_snapshot(
        {
            "run_id": "run-ui",
            "thread_id": "thread-ui",
            "stage": "Scoping",
            "scope_assessment": {
                "need_clarification": True,
                "question": "Which workload?",
                "verification": "",
            },
            "activity": [
                {
                    "id": "clarify",
                    "label": "Evaluating scope",
                    "agent": "Scope Agent",
                    "status": "success",
                },
            ],
        }
    )

    text = _visible_text(app)
    assert "I want to" not in text  # no brief text was ever produced
    assert 'Scope Agent</span><span class="ic-chip ic-run">Running</span>' in text
    # Planner hasn't started yet — its row stays hidden rather than showing as "Pending".
    assert "Planner" not in text


def test_planner_section_lists_todo_questions_as_bullets_once_complete():
    app = _app_with_snapshot(
        {
            "run_id": "run-ui",
            "thread_id": "thread-ui",
            "stage": "Researching",
            "todos": [
                {
                    "todo_id": "todo-1",
                    "research_question": "Find primary sources",
                    "status": "planned",
                },
                {"todo_id": "todo-2", "research_question": "Compare findings", "status": "planned"},
            ],
            "activity": [
                {
                    "id": "plan",
                    "label": "Calling write_todos",
                    "agent": "Planner",
                    "status": "success",
                },
                {
                    "id": "research",
                    "label": "Delegating research",
                    "agent": "Research Supervisor",
                    "status": "running",
                },
            ],
        }
    )

    text = _visible_text(app)
    assert (
        text.index("Planner")
        < text.index("Find primary sources")
        < text.index("Compare findings")
        < text.index("Researcher")
    )
    assert 'Planner</span><span class="ic-chip ic-ok">Complete</span>' in text


def test_researcher_section_lists_tavily_sources_for_each_completed_todo(tmp_path, monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_VFS_ROOT", str(tmp_path))
    vfs = VFS(str(tmp_path))
    vfs.write_json(
        "run-ui",
        "research/todo-1/evidence/result.json",
        {
            "sources": [
                {"locator": "https://example.com/a", "provenance": {"title": "Source A"}},
                {"locator": "https://example.com/b", "provenance": {"title": "Source B"}},
            ]
        },
        "evidence_bundle",
        task_id="todo-1",
    )

    app = _app_with_snapshot(
        {
            "run_id": "run-ui",
            "thread_id": "thread-ui",
            "stage": "Researching",
            "todos": [
                {
                    "todo_id": "todo-1",
                    "research_question": "Find primary sources",
                    "status": "completed",
                },
            ],
            "activity": [
                {
                    "id": "research",
                    "label": "Delegating research",
                    "agent": "Research Supervisor",
                    "status": "success",
                },
                {
                    "id": "draft",
                    "label": "Drafting report",
                    "agent": "Report Writer",
                    "status": "running",
                },
            ],
        }
    )

    assert not app.exception
    text = _visible_text(app)
    assert text.index("Researcher") < text.index("Synthesizer")
    # The question is the expander's own label (not separate markdown text); its position
    # among expanders — before the unrelated "Details" expander — confirms it belongs to
    # the Researcher row, which renders first.
    assert app.expander[0].label == "Find primary sources"
    question_expander = app.expander[0]
    expander_text = "\n".join(item.value for item in question_expander.markdown)
    assert "Source A" in expander_text
    assert "https://example.com/a" in expander_text
    assert "Source B" in expander_text
