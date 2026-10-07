import json

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from intelligence_collector.agents.planner import PlannerAgent
from intelligence_collector.agents.tools import WRITE_TODOS_TOOL_NAME
from intelligence_collector.contracts import ResearchBrief
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.storage.vfs import VFS

BRIEF = ResearchBrief(
    original_question="Which Python version is newest?",
    research_question="Find the newest stable Python release and its release date.",
)
QUESTION = (
    "What is the newest stable CPython release listed on python.org, and when was it released?"
)


class ToolCallingPlanner:
    """Tool-calling fake: replays scripted ``write_todos`` arguments, one per call."""

    model_name = "fake-planner"

    def __init__(self, calls):
        self.calls = list(calls)
        self.bound = []
        self.prompts = []

    def bind_tools(self, tools, tool_choice=None):
        self.bound.append(([tool["function"]["name"] for tool in tools], tool_choice))
        return self

    def invoke(self, messages, config=None):
        self.prompts.append(list(messages))
        args = self.calls.pop(0)
        return AIMessage(
            content="",
            tool_calls=[
                {"name": WRITE_TODOS_TOOL_NAME, "args": args, "id": f"c{len(self.prompts)}"}
            ],
        )


def _args(questions):
    return {
        "questions": questions,
        "evidence_needed": ["python.org release listing"] * len(questions),
        "evidence_points": [[{"text": "newest stable version and date", "required_answers": [1]}]]
        * len(questions),
        "report_shape": "A short direct answer, then sources.",
    }


def test_planner_records_todos_through_write_todos_tool_call(tmp_path):
    model = ToolCallingPlanner([_args([QUESTION])])
    vfs = VFS(tmp_path)

    todos, shape = PlannerAgent(ModelRouter(model)).plan(BRIEF, vfs, "run")

    assert model.bound == [([WRITE_TODOS_TOOL_NAME], WRITE_TODOS_TOOL_NAME)]
    assert [todo.research_question for todo in todos] == [QUESTION]
    assert todos[0].evidence_needed == "python.org release listing"
    assert shape == "A short direct answer, then sources."
    persisted = json.loads((tmp_path / "run" / "planning" / "todos.v1.json").read_text())
    assert [todo["research_question"] for todo in persisted] == [QUESTION]


def test_invalid_write_todos_call_is_answered_with_the_error_and_retried(tmp_path):
    model = ToolCallingPlanner([_args([QUESTION, QUESTION.lower()]), _args([QUESTION])])

    plan = PlannerAgent(ModelRouter(model)).define_plan(BRIEF)

    assert plan.questions == [QUESTION]
    retry = model.prompts[1]
    assert isinstance(retry[-1], ToolMessage) and retry[-1].tool_call_id == "c1"
    assert "must not repeat" in retry[-1].content


def test_planner_fails_after_repeated_invalid_write_todos_calls(tmp_path):
    model = ToolCallingPlanner([_args([]), _args([])])
    with pytest.raises(ValueError, match="invalid structured output for planner"):
        PlannerAgent(ModelRouter(model)).plan(BRIEF, VFS(tmp_path), "run")
    assert not (tmp_path / "run" / "planning" / "todos.v1.json").exists()


TWO_ANSWERS = ResearchBrief(
    original_question="Compare SQLite size limits and licences.",
    research_question="Compare the maximum database size and the licence of SQLite.",
    required_answers=["maximum database size", "licence"],
)


def _linked(*links):
    """write_todos args for one question whose evidence points link to `links` (one per point)."""
    return {
        "questions": [QUESTION],
        "evidence_needed": [""],
        "evidence_points": [
            [{"text": f"point {i}", "required_answers": link} for i, link in enumerate(links)]
        ],
        "report_shape": "Short answer.",
    }


def test_unlinked_required_answer_is_retried_with_the_missing_item(tmp_path):
    model = ToolCallingPlanner([_linked([1]), _linked([1], [2])])

    todos, _ = PlannerAgent(ModelRouter(model)).plan(TWO_ANSWERS, VFS(tmp_path), "run")

    retry = model.prompts[1][-1]
    assert isinstance(retry, ToolMessage) and "unlinked: licence" in retry.content
    points = todos[0].evidence_points
    assert [(p.point_id, p.required_answers) for p in points] == [
        ("E1", ["maximum database size"]),
        ("E2", ["licence"]),
    ]
    assert todos[0].evidence_needed == "point 0; point 1"


def test_out_of_range_link_and_five_points_are_retried(tmp_path):
    model = ToolCallingPlanner(
        [_linked([3], [1, 2]), _linked(*[[1, 2]] * 5), _linked([1, 2])]
    )

    plan = PlannerAgent(ModelRouter(model, max_attempts=3)).define_plan(TWO_ANSWERS)

    assert "outside 1..2" in model.prompts[1][-1].content
    assert "1 to 4 evidence points" in model.prompts[2][-1].content
    assert len(plan.evidence_points[0]) == 1


def test_plan_hash_covers_evidence_points_but_not_follow_ups():
    from intelligence_collector.agents.planner import plan_record
    from intelligence_collector.contracts import EvidencePoint, ResearchTodo, TaskOrigin

    def todo(text, **extra):
        return ResearchTodo(
            todo_id="t1",
            research_question=QUESTION,
            evidence_points=[EvidencePoint(point_id="E1", text=text, required_answers=["a"])],
            **extra,
        )

    base = plan_record(TWO_ANSWERS, [todo("version")], "short")
    changed = plan_record(TWO_ANSWERS, [todo("release date")], "short")
    follow_up = ResearchTodo(
        todo_id="f1", research_question="Follow-up?", origin=TaskOrigin.FOLLOW_UP, round=2,
        evidence_points=[EvidencePoint(point_id="E1", text="gap", required_answers=["a"])],
        gap_point_ids=["t1:E1"],
    )
    with_follow_up = plan_record(TWO_ANSWERS, [todo("version"), follow_up], "short")

    assert base["tasks"][0]["evidence_points"][0]["text"] == "version"
    assert changed["plan_hash"] != base["plan_hash"]
    assert with_follow_up["plan_hash"] == base["plan_hash"]
    assert [f["todo_id"] for f in with_follow_up["follow_ups"]] == ["f1"]
    assert with_follow_up["follow_ups"][0]["gap_point_ids"] == ["t1:E1"]
