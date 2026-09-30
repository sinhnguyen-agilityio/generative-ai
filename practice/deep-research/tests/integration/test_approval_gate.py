"""Integration tests for the human-in-the-loop review, reached once a report has fully passed
verification: evaluating_faithfulness -> writer -> approval (HITL).

* Approve goes to ``publishing``, which writes the approved text to ``output_dir`` and ends.
* Reject hands off to the ``classify_feedback`` node, whose feedback-router LLM decides between a
  formatting request (back to ``writer``, which only re-presents the report — no re-synthesis, no
  re-verification — then back to ``approval``) or a scope change (back to ``clarify_with_user``,
  re-entering the full Build-Verify-Fix loop).
"""

import hashlib
import json

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from intelligence_collector.agents.feedback_router import FeedbackRouter
from intelligence_collector.agents.planner import PlannerAgent
from intelligence_collector.agents.report_judge import ReportJudge
from intelligence_collector.agents.report_writer import ReportWriter
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.contracts import ResearchTodo
from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.nodes import Node, ResearchNodes
from intelligence_collector.graph.routing import (
    APPROVED,
    REJECTED,
    TO_FORMAT,
    TO_PUBLISH,
    TO_SCOPE,
    route_after_approval,
    route_after_feedback,
)
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.fake_models import FakeAcceptanceModel, FakeReportModel, FakeResearchModel, cited_report
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
BULLETED = GOOD.replace("A cache stores", "- A cache stores")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _single_question_plan():
    return {
        "questions": [QUESTION],
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


def _passing_judge() -> ReportJudge:
    """Every draft this suite writes is already well-formed; only the approval gate itself is
    under test here, so the judge always reports every criterion met."""
    model = FakeAcceptanceModel([{
        "results": [{"criterion": "all criteria", "met": True, "justification": "Satisfied."}],
    }])
    return ReportJudge(ModelRouter(model))


def _bvf_graph(stores, report_model, *, output_dir, feedback_router=None):
    """A graph with a real (fake-backed) BVF loop but no scope_agent — todos/brief are injected
    directly into the initial state, exactly like tests/integration/test_bvf_loop.py."""
    vfs, index = stores
    researcher = ResearchAgent(
        ModelRouter(FakeResearchModel()),
        lambda: ControlledMCP([{"url": URL, "content": "A cache stores reusable data for later access."}]),
    )
    nodes = ResearchNodes(
        vfs, index, researcher,
        report_writer=ReportWriter(ModelRouter(report_model)),
        report_judge=_passing_judge(),
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


def test_reject_with_format_route_only_reformats_then_publishes_the_approved_text(stores, tmp_path):
    report_model = FakeReportModel([GOOD, BULLETED])
    feedback_model = FakeFeedbackRouterModel([{"route": "format", "reason": "Just formatting."}])
    graph = _bvf_graph(
        stores, report_model,
        feedback_router=FeedbackRouter(ModelRouter(feedback_model)),
        output_dir=tmp_path / "exports",
    )
    config = {"configurable": {"thread_id": "approval-run"}}

    paused = graph.invoke(_initial_state(), config)
    verification_id = paused["verification"]["verification_id"]
    paused_again = graph.invoke(
        Command(resume={"approved": False, "feedback": "Please use bullet points."}), config,
    )

    assert paused_again.get("__interrupt__")
    assert paused_again["stage"] == "Awaiting Approval"
    assert paused_again["feedback_route"] == "format"
    assert paused_again["human_feedback"] == ""  # consumed by writer
    assert feedback_model.calls
    write_prompts = report_model.prompts("write")
    assert len(write_prompts) == 2  # the first draft, then one formatting edit — no re-synthesis
    assert "Please use bullet points." in write_prompts[1]
    assert "report formatter" in write_prompts[1]
    assert "takes precedence" in write_prompts[1]
    # Presented in the new format, without re-verification: still the original verification.
    assert paused_again["draft"] == BULLETED
    assert paused_again["verified_draft"] == GOOD
    assert paused_again["verification"]["verification_id"] == verification_id
    assert paused_again["verification"]["artifact_hash"] == _sha(GOOD)
    assert paused_again["format_revision"] == 1
    assert paused_again["format_unchanged"] is False
    human_entries = [e for e in paused_again["bvf_history"] if e["issue_type"] == "HUMAN_FEEDBACK"]
    assert human_entries and human_entries[0]["route"] == "format"

    result = graph.invoke(Command(resume={"approved": True}), config)
    assert result["stage"] == "Published"
    # The reviewer approved the reformatted text, so that exact text is what gets published.
    assert (tmp_path / "exports" / "approval-run.md").read_text(encoding="utf-8") == BULLETED


def test_format_request_that_changes_nothing_is_reported(stores, tmp_path):
    feedback_model = FakeFeedbackRouterModel([{"route": "format", "reason": "Just formatting."}])
    graph = _bvf_graph(
        stores, FakeReportModel([GOOD, GOOD]),
        feedback_router=FeedbackRouter(ModelRouter(feedback_model)),
        output_dir=tmp_path / "exports",
    )
    config = {"configurable": {"thread_id": "approval-run"}}

    graph.invoke(_initial_state(), config)
    paused_again = graph.invoke(
        Command(resume={"approved": False, "feedback": "Please use bullet points."}), config,
    )

    assert paused_again["format_unchanged"] is True
    assert "left the report unchanged" in paused_again["current_action"]


def test_review_flow_graph_structure(stores, tmp_path):
    """faithfulness -> writer -> approval (HITL); approve -> publishing; reject -> classify_feedback
    (LLM) -> writer (format) or scoping (scope)."""
    graph = _bvf_graph(stores, FakeReportModel([GOOD]), output_dir=tmp_path / "exports")
    edges = {(e.source, e.target) for e in graph.get_graph().edges}

    assert (Node.EVALUATE_FAITHFULNESS, Node.WRITER) in edges
    assert (Node.WRITER, Node.APPROVE) in edges
    assert (Node.APPROVE, Node.PUBLISH) in edges
    assert (Node.PUBLISH, "__end__") in edges
    assert (Node.APPROVE, Node.CLASSIFY_FEEDBACK) in edges
    assert (Node.CLASSIFY_FEEDBACK, Node.WRITER) in edges
    assert (Node.CLASSIFY_FEEDBACK, Node.PUBLISH) in edges  # a chat reply read as an approval
    assert (Node.CLASSIFY_FEEDBACK, Node.STUB_SCOPE) in edges  # clarify_with_user when a scope agent is wired
    assert (Node.CLASSIFY_FEEDBACK, Node.SYNTHESIZE) not in edges  # formatting never re-synthesizes


def test_feedback_routing_outcomes():
    assert route_after_approval({"approval": {"approved": True}}) == APPROVED
    assert route_after_approval({"approval": {"approved": False}}) == REJECTED
    assert route_after_feedback({"feedback_route": "scope"}) == TO_SCOPE
    assert route_after_feedback({"feedback_route": "format"}) == TO_FORMAT
    assert route_after_feedback({"feedback_route": "approve", "approval": {"approved": True}}) == TO_PUBLISH
    # An "approve" route without a recorded approval never publishes.
    assert route_after_feedback({"feedback_route": "approve", "approval": None}) == TO_FORMAT


def test_reject_without_a_feedback_router_defaults_to_format(stores, tmp_path):
    report_model = FakeReportModel([GOOD, GOOD])
    graph = _bvf_graph(stores, report_model, feedback_router=None, output_dir=tmp_path / "exports")
    config = {"configurable": {"thread_id": "approval-run"}}

    graph.invoke(_initial_state(), config)
    paused_again = graph.invoke(
        Command(resume={"approved": False, "feedback": "not great"}), config,
    )

    assert paused_again["feedback_route"] == "format"


def test_reject_with_scope_route_goes_back_through_scoping_and_replans(stores, tmp_path):
    vfs, index = stores
    shared_model = RecordingChatModel([
        _guardrail_payload(), _clear_payload(), _brief_payload(),
        _single_question_plan(),  # initial plan
        _guardrail_payload(), _clear_payload(), _brief_payload(),  # feedback re-enters scoping
        _single_question_plan(),  # replan after the scope change
    ])
    shared_router = ModelRouter(shared_model)
    feedback_model = FakeFeedbackRouterModel([{"route": "scope", "reason": "Changes what is covered."}])

    researcher = ResearchAgent(
        ModelRouter(FakeResearchModel()),
        lambda: ControlledMCP([{"url": URL, "content": "A cache stores reusable data for later access."}]),
    )
    report_model = FakeReportModel([GOOD, GOOD])
    nodes = ResearchNodes(
        vfs, index, researcher,
        report_writer=ReportWriter(ModelRouter(report_model)),
        report_judge=_passing_judge(),
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
        feedback_router=FeedbackRouter(ModelRouter(feedback_model)),
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
        Command(resume={"approved": False, "feedback": "Also cover the wrong angle: eviction policies."}),
        config,
    )

    assert paused_again.get("__interrupt__")
    assert paused_again["stage"] == "Awaiting Approval"
    assert paused_again["feedback_route"] == "scope"
    assert paused_again["human_feedback"] == ""
    assert paused_again["todos"]
    human_entries = [e for e in paused_again["bvf_history"] if e["issue_type"] == "HUMAN_FEEDBACK"]
    assert human_entries and human_entries[0]["route"] == "scope"
    # The feedback re-entered scoping as a new user turn, before the brief and plan were redone.
    scope_prompts = ["\n".join(str(m.content) for m in call) for call in shared_model.calls[4:]]
    assert any("eviction policies" in prompt for prompt in scope_prompts)
    assert paused_again["recovery_attempt"] == 0  # a new scope starts a fresh Build-Verify-Fix cycle

    result = graph.invoke(Command(resume={"approved": True}), config)

    assert result["stage"] == "Published"
    assert (tmp_path / "exports" / "planner-run.md").exists()


def test_chat_reply_classified_as_approve_publishes(stores, tmp_path):
    feedback_model = FakeFeedbackRouterModel([{"route": "approve", "reason": "Accepts as-is."}])
    graph = _bvf_graph(
        stores, FakeReportModel([GOOD]),
        feedback_router=FeedbackRouter(ModelRouter(feedback_model)),
        output_dir=tmp_path / "exports",
    )
    config = {"configurable": {"thread_id": "approval-run"}}

    graph.invoke(_initial_state(), config)
    result = graph.invoke(Command(resume={"reply": "Looks good, publish it", "actor": "tester"}), config)

    assert result["stage"] == "Published"
    assert result["approval"]["approved"] is True
    assert result["approval"]["actor"] == "tester"
    assert (tmp_path / "exports" / "approval-run.md").read_text(encoding="utf-8") == GOOD
    # The review reads as chat: the reviewer's reply, then the agent's outcome.
    chat = [(m.type, m.content) for m in result["messages"]]
    assert chat[-2] == ("human", "Looks good, publish it")
    assert chat[-1][0] == "ai" and chat[-1][1].startswith("Published to")


def test_chat_reply_classified_as_format_reformats_and_waits_again(stores, tmp_path):
    feedback_model = FakeFeedbackRouterModel([{"route": "format", "reason": "Formatting."}])
    graph = _bvf_graph(
        stores, FakeReportModel([GOOD, BULLETED]),
        feedback_router=FeedbackRouter(ModelRouter(feedback_model)),
        output_dir=tmp_path / "exports",
    )
    config = {"configurable": {"thread_id": "approval-run"}}

    graph.invoke(_initial_state(), config)
    paused_again = graph.invoke(Command(resume={"reply": "Use bullet points"}), config)

    assert paused_again.get("__interrupt__")
    assert paused_again["draft"] == BULLETED
    assert paused_again["approval"] is None  # a chat reply is not a decision until classified
    chat = [(m.type, m.content) for m in paused_again["messages"]]
    assert chat[-2] == ("human", "Use bullet points")
    assert chat[-1][0] == "ai" and "reformatted" in chat[-1][1]


def test_explicit_rejection_is_never_classified_as_approval(stores, tmp_path):
    feedback_model = FakeFeedbackRouterModel([{"route": "approve", "reason": "Sounds positive."}])
    graph = _bvf_graph(
        stores, FakeReportModel([GOOD, BULLETED]),
        feedback_router=FeedbackRouter(ModelRouter(feedback_model)),
        output_dir=tmp_path / "exports",
    )
    config = {"configurable": {"thread_id": "approval-run"}}

    graph.invoke(_initial_state(), config)
    paused_again = graph.invoke(
        Command(resume={"approved": False, "feedback": "Great, but use bullets"}), config,
    )

    assert paused_again["stage"] == "Awaiting Approval"
    assert paused_again["feedback_route"] == "format"
    assert not (tmp_path / "exports" / "approval-run.md").exists()
