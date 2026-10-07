import json

import pytest
from langchain_core.messages import AIMessage

from intelligence_collector.agents.planner import PlannerAgent, plan_record
from intelligence_collector.contracts import (
    EvidencePoint,
    ResearchBrief,
    ResearchTodo,
    TaskOrigin,
)
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.storage.vfs import VFS

BRIEF = ResearchBrief(
    original_question="Which Python version is newest?",
    research_question="Find the newest stable Python release and its release date.",
)
QUESTION = (
    "What is the newest stable CPython release listed on python.org, and when was it released?"
)
TWO_ANSWERS = ResearchBrief(
    original_question="Compare SQLite size limits and licences.",
    research_question="Compare the maximum database size and the licence of SQLite.",
    required_answers=["maximum database size", "licence"],
)


class JsonPlanner:
    """Fake model: replays scripted plans as JSON content, one per call."""

    model_name = "fake-planner"

    def __init__(self, plans):
        self.plans = list(plans)
        self.prompts = []

    def invoke(self, messages, config=None):
        self.prompts.append(list(messages))
        return AIMessage(content=json.dumps(self.plans.pop(0)))


def _plan(questions, links=([1],)):
    return {
        "questions": questions,
        "evidence_needed": ["python.org release listing"] * len(questions),
        "evidence_points": [
            [{"text": f"point {i}", "required_answers": link} for i, link in enumerate(links)]
        ]
        * len(questions),
        "report_shape": "A short direct answer, then sources.",
    }


def test_planner_saves_generated_todos_to_vfs(tmp_path):
    model = JsonPlanner([_plan([QUESTION])])

    todos, shape = PlannerAgent(ModelRouter(model)).plan(BRIEF, VFS(tmp_path), "run")

    assert [todo.research_question for todo in todos] == [QUESTION]
    assert todos[0].evidence_needed == "python.org release listing"
    assert shape == "A short direct answer, then sources."
    persisted = json.loads((tmp_path / "run" / "planning" / "todos.v1.json").read_text())
    assert [todo["research_question"] for todo in persisted] == [QUESTION]


def test_prompt_carries_feedback_and_max_todos(tmp_path):
    model = JsonPlanner([_plan([QUESTION]), _plan([QUESTION])])
    planner = PlannerAgent(ModelRouter(model))

    planner.plan(BRIEF, VFS(tmp_path), "run", max_todos=2)
    planner.plan(BRIEF, VFS(tmp_path), "run", feedback="Only use python.org.")

    without, with_feedback = (prompt[-1].content for prompt in model.prompts)
    assert "at most 2" in model.prompts[0][0].content
    assert "Human feedback" not in without
    assert "Human feedback" in with_feedback and "Only use python.org." in with_feedback


def test_more_tasks_than_max_todos_is_rejected(tmp_path):
    model = JsonPlanner([_plan([QUESTION, "A second, different question?"])])

    with pytest.raises(ValueError, match="max is 1"):
        PlannerAgent(ModelRouter(model)).plan(BRIEF, VFS(tmp_path), "run", max_todos=1)
    assert not (tmp_path / "run" / "planning" / "todos.v1.json").exists()


def test_invalid_plan_is_retried_then_fails_without_writing(tmp_path):
    model = JsonPlanner([_plan([QUESTION, QUESTION.lower()]), _plan([])])

    with pytest.raises(ValueError, match="invalid structured output for planner"):
        PlannerAgent(ModelRouter(model)).plan(BRIEF, VFS(tmp_path), "run")
    assert len(model.prompts) == 2
    assert not (tmp_path / "run" / "planning" / "todos.v1.json").exists()


def test_evidence_point_links_resolve_to_required_answers(tmp_path):
    model = JsonPlanner([_plan([QUESTION], links=([1], [2]))])

    todos, _ = PlannerAgent(ModelRouter(model)).plan(TWO_ANSWERS, VFS(tmp_path), "run")

    assert [(p.point_id, p.required_answers) for p in todos[0].evidence_points] == [
        ("E1", ["maximum database size"]),
        ("E2", ["licence"]),
    ]


def test_unlinked_required_answer_is_rejected_unless_revising_from_feedback(tmp_path):
    with pytest.raises(ValueError, match="without an evidence point: licence"):
        PlannerAgent(ModelRouter(JsonPlanner([_plan([QUESTION])]))).plan(
            TWO_ANSWERS, VFS(tmp_path), "run"
        )
    assert not (tmp_path / "run" / "planning" / "todos.v1.json").exists()

    todos, _ = PlannerAgent(ModelRouter(JsonPlanner([_plan([QUESTION])]))).plan(
        TWO_ANSWERS, VFS(tmp_path), "run", feedback="Plan only for the maximum database size."
    )
    assert todos[0].evidence_points[0].required_answers == ["maximum database size"]


def test_out_of_range_link_is_rejected(tmp_path):
    model = JsonPlanner([_plan([QUESTION], links=([3],))])

    with pytest.raises(ValueError, match="outside 1..2"):
        PlannerAgent(ModelRouter(model)).plan(TWO_ANSWERS, VFS(tmp_path), "run")


def test_plan_hash_covers_evidence_points_but_not_follow_ups():
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
