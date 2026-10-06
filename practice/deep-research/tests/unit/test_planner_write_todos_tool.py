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
