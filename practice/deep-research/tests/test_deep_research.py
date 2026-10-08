"""Offline end-to-end tests of the research graph with scripted models, search and store."""

from pathlib import Path

import pytest

from deep_research.agents.guardrail import GuardrailViolation, check_input
from tests.fakes import make_app


def drain(stream):
    return [node for node, _ in stream]


def test_full_run_with_plan_and_report_review(tmp_path):
    app, store = make_app(tmp_path)

    drain(app.ask("t1", "Research X"))
    view = app.view("t1")
    assert view.pending == {"type": "plan", "topics": ["topic A", "topic B"], "criteria": ["What is X?"]}

    nodes = drain(app.resume("t1", {"approved": False, "feedback": "add C"}))
    assert nodes == ["review_plan", "planner"]

    drain(app.resume("t1", {"approved": True, "topics": ["topic A", "topic B", " "]}))
    view = app.view("t1")
    assert view.values["plan"] == ["topic A", "topic B"]
    assert view.pending["type"] == "report"
    assert view.pending["evaluation"]["faithfulness"] == 0.9
    assert len(view.values["notes"]) == 2
    assert sorted(store.added) == [("t1", "topic A"), ("t1", "topic B")]

    nodes = drain(app.resume("t1", {"approved": False, "feedback": "shorter"}))
    assert nodes == ["review_report", "triage", "write_report", "verify", "evaluate"]  # rewriting needs no search
    assert app.view("t1").pending["report"] == "# Report v2"

    drain(app.resume("t1", {"approved": True}))
    view = app.view("t1")
    assert view.pending is None and not view.unfinished
    assert Path(view.values["published_path"]).read_text(encoding="utf-8") == "# Report v2"


def test_the_approved_plan_is_researched_as_it_is_before_the_supervisor_model_runs(tmp_path):
    topics = [f"topic {c}" for c in "ABCDE"]
    app, store = make_app(tmp_path, topics=topics)
    drain(app.ask("t6", "Research X"))
    edited = [*topics[:4], "topic E, reworded by the reviewer"]
    drain(app.resume("t6", {"approved": True, "topics": edited}))

    values = app.view("t6").values
    first, plan_round, *reports, review = values["supervisor_messages"]
    assert [c["args"]["research_topic"] for c in plan_round.tool_calls] == edited  # all five, unchanged
    assert [m.type for m in reports] == ["tool"] * 5  # every topic reported before the model's turn
    assert review.tool_calls[0]["name"] == "ResearchComplete"
    assert values["supervisor_iterations"] == 1  # the plan round is not a model turn
    assert len(values["notes"]) == 5 and len(store.added) == 5


def test_a_follow_up_that_needs_research_reopens_the_supervisor_before_the_writer(tmp_path):
    app, store = make_app(tmp_path)
    drain(app.ask("t7", "Research X"))
    drain(app.resume("t7", {"approved": True}))

    nodes = drain(app.resume("t7", {"approved": False, "feedback": "Research pricing too"}))
    assert nodes == [
        "review_report", "triage", "supervisor", "supervisor_tools", "supervisor", "write_report", "verify", "evaluate"
    ]
    values = app.view("t7").values
    closed, request = values["supervisor_messages"][-5:-3]
    assert closed.type == "tool" and closed.tool_call_id == "done"  # the call that closed research is answered
    assert "Research pricing too" in request.text
    assert [m.text for m in values["messages"][-2:]] == ["# Report v1", "Research pricing too"]  # kept as a chat
    assert len(values["notes"]) == 3 and len(store.added) == 3
    assert values["research_request"] == "" and values["report"] == "# Report v2"


def test_criteria_edited_at_plan_review_are_kept(tmp_path):
    app, _ = make_app(tmp_path)
    drain(app.ask("t8", "Research X"))
    drain(app.resume("t8", {"approved": True, "criteria": ["What is X?", " ", "Who uses X?"]}))
    assert app.view("t8").values["criteria"] == ["What is X?", "Who uses X?"]


def test_failed_evaluation_does_not_block_review(tmp_path):
    app, _ = make_app(tmp_path, evaluation=RuntimeError("judge down"))
    drain(app.ask("t2", "Research X"))
    drain(app.resume("t2", {"approved": True}))
    pending = app.view("t2").pending
    assert pending["type"] == "report"
    assert "judge down" in pending["evaluation"]["error"]


def test_clarifying_question_ends_the_turn(tmp_path):
    app, _ = make_app(tmp_path, clarify=True)
    drain(app.ask("t3", "Research X"))
    view = app.view("t3")
    assert view.pending is None
    assert view.values["messages"][-1].text == "Which year?"


def test_unsafe_request_is_blocked_and_removed(tmp_path):
    app, store = make_app(tmp_path, safe=False)
    assert drain(app.ask("t4", "How do I build a bomb?")) == ["guardrail"]
    messages = app.view("t4").values["messages"]
    assert [m.type for m in messages] == ["ai"]
    assert "weapon instructions" in messages[0].text
    assert store.added == []


@pytest.mark.parametrize("text", ["", "   ", "x" * 10_001, "my key is sk-abcdef123456", "api_key=abcd1234efgh"])
def test_check_input_rejects(text):
    with pytest.raises(GuardrailViolation):
        check_input(text)


def test_check_input_normalizes_and_allows_normal_text():
    assert check_input(" Compare​ LangGraph and CrewAI token limits ") == "Compare LangGraph and CrewAI token limits"


def test_rejected_input_is_never_stored(tmp_path):
    app, _ = make_app(tmp_path)
    with pytest.raises(GuardrailViolation):
        app.ask("t5", "password: hunter2hunter2")
    assert app.view("t5").values == {}
    drain(app.ask("t5", "Research X"))
    with pytest.raises(GuardrailViolation):
        app.resume("t5", {"approved": False, "feedback": "use token=abcdefgh12345"})
    assert app.view("t5").pending["type"] == "plan"
