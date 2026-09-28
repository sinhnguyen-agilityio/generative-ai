"""Integration tests for the human-in-the-loop approval gate (``ApprovalNodes``), reached once
a report has fully passed verification:

* Approve writes the exact verified artifact to ``output_dir`` and ends the run.
* Reject, with feedback, is classified by a feedback-router LLM as needing either a report-writer
  revision (routes back to ``synthesizing``) or a full replan (routes back to ``planning``) —
  either path re-enters the normal Build-Verify-Fix loop and can reach this same gate again.
"""

import json

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from intelligence_collector.agents.feedback_router import FeedbackRouter
from intelligence_collector.agents.planner import PlannerAgent
from intelligence_collector.agents.report_writer import ReportWriter
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.contracts import ResearchTodo
from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.nodes import ResearchNodes
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.fake_models import FakeReportModel, FakeResearchModel, cited_report
from tests.integration.test_langchain_scope import (
    RecordingChatModel,
    _brief_payload,
    _clear_payload,
    _guardrail_payload,
)

URL = "https://controlled.test/cache"
QUESTION = "How does a local software cache store and reuse data?"
BRIEF = {"research_question": "I want a technical report on local software cache behavior from primary sources."}
GOOD = cited_report(["A cache stores reusable data for later access [S1]."], {"S1": URL})


def _single_question_plan():
    return {
        "questions": [QUESTION],
        "acceptance_criteria": ["States how the cache stores and reuses data."],
    }


class FakeFeedbackRouterModel:
    """Scripted feedback-router fake; returns queued route decisions in call order."""

    model_name = "fake-feedback-router"

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        decision = self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]
        return AIMessage(content=json.dumps(decision))


def _bvf_graph(stores, report_model, *, output_dir, feedback_router=None):
    """A graph with a real (fake-backed) BVF loop but no scope_agent — todos/brief are injected
    directly into the initial state, exactly like tests/integration/test_bvf_loop.py."""
    vfs, index = stores
    researcher = ResearchAgent(
        ModelRouter(FakeResearchModel(), mode="fake"),
        lambda: ControlledMCP([{"url": URL, "content": "A cache stores reusable data for later access."}]),
    )
    nodes = ResearchNodes(
        vfs, index, researcher,
        report_writer=ReportWriter(ModelRouter(report_model, mode="fake")),
        recovery_budget=3,
    )
    return build_graph(
        research_nodes=nodes,
        vfs=vfs,
        checkpointer=MemorySaver(),
        with_approval=True,
        output_dir=str(output_dir),
        feedback_router=feedback_router,
    )


def _initial_state(run_id="approval-run"):
    todo = ResearchTodo(todo_id="todo-1", research_question=QUESTION)
    return {
        "run_id": run_id,
        "thread_id": run_id,
        "scoped_request": BRIEF,
        "todos": [todo.model_dump(mode="json")],
    }


def test_approve_publishes_the_verified_report_to_output_dir(stores, tmp_path):
    graph = _bvf_graph(stores, FakeReportModel([GOOD]), output_dir=tmp_path / "exports")
    config = {"configurable": {"thread_id": "approval-run"}}

    paused = graph.invoke(_initial_state(), config)
    assert paused.get("__interrupt__")
    assert paused["stage"] == "Awaiting Approval"

    result = graph.invoke(Command(resume={"approved": True, "actor": "tester"}), config)

    assert result["stage"] == "Published"
    assert result["publication"]["published"] is True
    published = tmp_path / "exports" / "approval-run.md"
    assert published.exists()
    assert published.read_text(encoding="utf-8") == GOOD


def test_reject_with_writer_route_revises_the_report_then_can_be_approved(stores, tmp_path):
    report_model = FakeReportModel([GOOD, GOOD])
    feedback_model = FakeFeedbackRouterModel([{"route": "writer", "reason": "Just wording."}])
    graph = _bvf_graph(
        stores, report_model,
        feedback_router=FeedbackRouter(ModelRouter(feedback_model, mode="fake")),
        output_dir=tmp_path / "exports",
    )
    config = {"configurable": {"thread_id": "approval-run"}}

    graph.invoke(_initial_state(), config)
    paused_again = graph.invoke(
        Command(resume={"approved": False, "feedback": "Please tighten the wording."}), config,
    )

    assert paused_again.get("__interrupt__")
    assert paused_again["stage"] == "Awaiting Approval"
    assert paused_again["feedback_route"] == "writer"
    assert paused_again["human_feedback"] == ""  # consumed by synthesizing()
    assert feedback_model.calls
    write_prompts = report_model.prompts("write")
    assert len(write_prompts) == 2
    assert "Please tighten the wording." in write_prompts[1]
    human_entries = [e for e in paused_again["bvf_history"] if e["issue_type"] == "HUMAN_FEEDBACK"]
    assert human_entries and human_entries[0]["route"] == "writer"

    result = graph.invoke(Command(resume={"approved": True}), config)
    assert result["stage"] == "Published"


def test_reject_without_a_feedback_router_defaults_to_writer(stores, tmp_path):
    report_model = FakeReportModel([GOOD, GOOD])
    graph = _bvf_graph(stores, report_model, feedback_router=None, output_dir=tmp_path / "exports")
    config = {"configurable": {"thread_id": "approval-run"}}

    graph.invoke(_initial_state(), config)
    paused_again = graph.invoke(
        Command(resume={"approved": False, "feedback": "not great"}), config,
    )

    assert paused_again["feedback_route"] == "writer"


def test_reject_with_planner_route_triggers_a_full_replan(stores, tmp_path):
    vfs, index = stores
    shared_model = RecordingChatModel([
        _guardrail_payload(), _clear_payload(), _brief_payload(),
        _single_question_plan(),  # initial plan
        _single_question_plan(),  # replan after rejection
    ])
    shared_router = ModelRouter(shared_model, mode="fake")
    feedback_model = FakeFeedbackRouterModel([{"route": "planner", "reason": "Wrong direction."}])

    researcher = ResearchAgent(
        ModelRouter(FakeResearchModel(), mode="fake"),
        lambda: ControlledMCP([{"url": URL, "content": "A cache stores reusable data for later access."}]),
    )
    report_model = FakeReportModel([GOOD, GOOD])
    nodes = ResearchNodes(
        vfs, index, researcher,
        report_writer=ReportWriter(ModelRouter(report_model, mode="fake")),
        recovery_budget=3,
    )
    graph = build_graph(
        scope_agent=ScopeAgent(shared_router),
        planner_agent=PlannerAgent(shared_router),
        research_nodes=nodes,
        vfs=vfs,
        checkpointer=MemorySaver(),
        with_approval=True,
        output_dir=str(tmp_path / "exports"),
        feedback_router=FeedbackRouter(ModelRouter(feedback_model, mode="fake")),
    )
    config = {"configurable": {"thread_id": "planner-run"}}

    paused = graph.invoke(
        {"run_id": "planner-run", "thread_id": "planner-run",
         "request": "Compare local software caches"},
        config,
    )
    assert paused.get("__interrupt__")
    assert paused["stage"] == "Awaiting Approval"

    paused_again = graph.invoke(
        Command(resume={"approved": False, "feedback": "The research answered the wrong angle."}),
        config,
    )

    assert paused_again.get("__interrupt__")
    assert paused_again["stage"] == "Awaiting Approval"
    assert paused_again["feedback_route"] == "planner"
    assert paused_again["human_feedback"] == ""  # consumed by planning()
    assert paused_again["todos"]
    human_entries = [e for e in paused_again["bvf_history"] if e["issue_type"] == "HUMAN_FEEDBACK"]
    assert human_entries and human_entries[0]["route"] == "planner"
    # The replan call actually received the reviewer's guidance in its prompt.
    replan_call = shared_model.calls[-1]
    rendered = "\n".join(str(m.content) for m in replan_call)
    assert "wrong angle" in rendered

    result = graph.invoke(Command(resume={"approved": True}), config)

    assert result["stage"] == "Published"
    assert (tmp_path / "exports" / "planner-run.md").exists()
