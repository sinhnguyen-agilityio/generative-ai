"""Offline end-to-end tests of the research graph with scripted models, search and store."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import aiosqlite
import pytest

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deep_research.agents.supervisor import SupervisorAgent
from deep_research.config import ModelSpec, Settings
from deep_research.agents.guardrail import GuardrailViolation, check_input
from tests.fakes import FakeTier, call, make_app, supervisor


def drain(stream):
    """Consume a graph stream and collect emitted node names.

    Args:
        stream: Iterator yielding node/update pairs.

    Returns:
        Node names in execution order.
    """
    return [node for node, _ in stream]


def test_full_run_with_plan_and_report_review(tmp_path):
    """Exercise plan revision, approval, report rewriting, and Markdown publication."""
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


def test_supervisor_model_receives_and_delegates_the_approved_plan(tmp_path):
    """Verify edited plan topics reach the supervisor model and all delegated reports return before review."""
    topics = [f"topic {c}" for c in "ABCDE"]
    app, store = make_app(tmp_path, topics=topics)
    drain(app.ask("t6", "Research X"))
    edited = [*topics[:4], "topic E, reworded by the reviewer"]
    drain(app.resume("t6", {"approved": True, "topics": edited}))

    values = app.view("t6").values
    first, plan_round, *reports, review = values["supervisor_messages"]
    assert [c["args"]["research_topic"] for c in plan_round.tool_calls] == edited  # all five, unchanged
    assert [m.type for m in reports] == ["tool"] * 5  # every topic reported before the review turn
    assert review.tool_calls[0]["name"] == "ResearchComplete"
    assert values["supervisor_iterations"] == 2  # initial delegation and review are both model turns
    assert len(values["notes"]) == 5 and len(store.added) == 5


def test_a_follow_up_that_needs_research_reopens_the_supervisor_before_the_writer(tmp_path):
    """Verify that a follow up that needs research reopens the supervisor before the writer."""
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
    """Verify that criteria edited at plan review are kept."""
    app, _ = make_app(tmp_path)
    drain(app.ask("t8", "Research X"))
    drain(app.resume("t8", {"approved": True, "criteria": ["What is X?", " ", "Who uses X?"]}))
    assert app.view("t8").values["criteria"] == ["What is X?", "Who uses X?"]


def test_failed_evaluation_does_not_block_review(tmp_path):
    """Verify that failed evaluation does not block review."""
    app, _ = make_app(tmp_path, evaluation=RuntimeError("judge down"))
    drain(app.ask("t2", "Research X"))
    drain(app.resume("t2", {"approved": True}))
    pending = app.view("t2").pending
    assert pending["type"] == "report"
    assert "judge down" in pending["evaluation"]["error"]


def test_clarifying_question_ends_the_turn(tmp_path):
    """Verify clarification returns a question without starting plan review."""
    app, _ = make_app(tmp_path, clarify=True)
    drain(app.ask("t3", "Research X"))
    view = app.view("t3")
    assert view.pending is None
    assert view.values["messages"][-1].text == "Which year?"


def test_unsafe_request_is_blocked_and_removed(tmp_path):
    """Verify that unsafe request is blocked and removed."""
    app, store = make_app(tmp_path, safe=False)
    assert drain(app.ask("t4", "How do I build a bomb?")) == ["guardrail"]
    messages = app.view("t4").values["messages"]
    assert [m.type for m in messages] == ["ai"]
    assert "weapon instructions" in messages[0].text
    assert store.added == []


@pytest.mark.parametrize("text", ["", "   ", "x" * 10_001, "my key is sk-abcdef123456", "api_key=abcd1234efgh"])
def test_check_input_rejects(text):
    """Verify that check input rejects."""
    with pytest.raises(GuardrailViolation):
        check_input(text)


def test_check_input_normalizes_and_allows_normal_text():
    """Verify that check input normalizes and allows normal text."""
    assert check_input(" Compare​ LangGraph and CrewAI token limits ") == "Compare LangGraph and CrewAI token limits"


def test_rejected_input_is_never_stored(tmp_path):
    """Verify that rejected input is never stored."""
    app, _ = make_app(tmp_path)
    with pytest.raises(GuardrailViolation):
        app.ask("t5", "password: hunter2hunter2")
    assert app.view("t5").values == {}
    drain(app.ask("t5", "Research X"))
    with pytest.raises(GuardrailViolation):
        app.resume("t5", {"approved": False, "feedback": "use token=abcdefgh12345"})
    assert app.view("t5").pending["type"] == "plan"


def test_supervisor_resumes_failed_review_without_repeating_research(tmp_path):
    """Verify that supervisor resumes failed review without repeating research."""
    calls = 0

    def fail_once(messages):
        """Fail the second supervisor turn once, then resume normal scripted decisions."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Supervisor temporarily unavailable")
        return supervisor(messages)

    app, store = make_app(tmp_path, supervisor_model=fail_once)
    drain(app.ask("resume-supervisor", "Research X"))
    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        drain(app.resume("resume-supervisor", {"approved": True}))
    assert app.view("resume-supervisor").unfinished
    assert len(store.added) == 2

    drain(app.continue_run("resume-supervisor"))
    view = app.view("resume-supervisor")
    assert view.pending["type"] == "report"
    assert len(store.added) == len(view.values["notes"]) == 2
    assert calls == 3


def test_supervisor_stops_at_the_iteration_limit(tmp_path):
    """Verify that supervisor stops at the iteration limit."""
    turns = 0

    def review(messages):
        """Keep emitting Think calls so the supervisor must stop at its code limit."""
        nonlocal turns
        turns += 1
        return AIMessage("", tool_calls=[call("Think", {"reflection": "Reviewing evidence"}, f"think-{turns}")])

    app, _ = make_app(tmp_path, supervisor_model=review)
    drain(app.ask("agent-stop", "Research X"))
    drain(app.resume("agent-stop", {"approved": True}))
    assert turns == 6
    assert app.view("agent-stop").pending["type"] == "report"


def test_async_sqlite_checkpoint_preserves_follow_up_notes(tmp_path):
    """Verify that async sqlite checkpoint preserves follow up notes."""
    app, store = make_app(tmp_path)
    async def checkpoint():
        """Open a temporary async SQLite saver inside the application event loop."""
        return AsyncSqliteSaver(await aiosqlite.connect(str(tmp_path / "checkpoints.sqlite3")))

    app.graph.checkpointer = app.loop.run(checkpoint())
    try:
        drain(app.ask("sqlite", "Research X"))
        drain(app.resume("sqlite", {"approved": True}))
        drain(app.resume("sqlite", {"approved": False, "feedback": "Research pricing too"}))
        assert len(app.view("sqlite").values["notes"]) == len(store.added) == 3
    finally:
        app.loop.run(app.graph.checkpointer.conn.close())
        app.loop.close()


def test_supervisor_runs_all_delegated_topics_concurrently():
    """Verify all delegated topics overlap even when their count exceeds prompt concurrency guidance."""
    active = peak = 0

    async def research(topic):
        """Yield during fake topic research and measure overlapping researcher calls."""
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return {"report": topic, "sources": []}

    settings = Settings(reasoning=ModelSpec("fake"), standard=ModelSpec("fake"))
    agent = SupervisorAgent(FakeTier({}, tools=supervisor), SimpleNamespace(research=research), settings)
    topics = [f"topic-{i}" for i in range(5)]
    message = AIMessage("", tool_calls=[call("ConductResearch", {"research_topic": t}, t) for t in topics])
    result = asyncio.run(agent.supervisor_tools({"supervisor_messages": [message]}))
    assert peak == 5
    assert [note["report"] for note in result["notes"]] == topics
    assert [m.tool_call_id for m in result["supervisor_messages"]] == topics


def test_supervisor_without_initial_plan_thinks_before_delegating():
    """Verify the no-plan prompt and scripted Think-to-delegation-to-completion flow."""
    seen = []

    def model(messages):
        """Script planning with Think, topic delegation, and completion in separate turns."""
        seen.append(messages)
        if not isinstance(messages[-1], ToolMessage):
            return AIMessage("", tool_calls=[call("Think", {"reflection": "Research the main question"}, "plan")])
        if messages[-1].name == "Think":
            return AIMessage("", tool_calls=[call("ConductResearch", {"research_topic": "Research X"}, "research")])
        return AIMessage("", tool_calls=[call("ResearchComplete", {}, "done")])

    async def research(topic):
        """Return a topic note without performing external research."""
        return {"report": topic, "sources": []}

    settings = Settings(reasoning=ModelSpec("fake"), standard=ModelSpec("fake"))
    agent = SupervisorAgent(FakeTier({}, tools=model), SimpleNamespace(research=research), settings)
    result = asyncio.run(agent.graph.ainvoke({"brief": "Research X"}))
    assert "<Initial plan>\nNone." in seen[0][-1].text
    assert "first use Think" in seen[0][0].text
    assert result["notes"] == [{"report": "Research X", "sources": []}]
    assert result["supervisor_iterations"] == 3


@pytest.mark.parametrize("calls", [[], [call("ResearchComplete", {}, "done")]])
def test_supervisor_ends_on_model_completion(calls):
    """Verify that supervisor ends on model completion."""
    settings = Settings(reasoning=ModelSpec("fake"), standard=ModelSpec("fake"))
    agent = SupervisorAgent(FakeTier({}, tools=supervisor), None, settings)
    assert agent.next_step({"supervisor_messages": [AIMessage("", tool_calls=calls)]}) == "__end__"
