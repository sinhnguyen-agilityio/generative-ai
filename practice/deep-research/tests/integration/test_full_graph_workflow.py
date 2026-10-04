"""Full graph with scope, planning, plan approval, research and the three verification gates,
driven by deterministic fake models and a controlled MCP tool (workflow behavior only)."""

import json

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from intelligence_collector.agents.report_judge import ReportJudge
from intelligence_collector.agents.report_writer import ReportWriter
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.contracts import ResearchTodo
from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.nodes import ResearchNodes
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.fake_models import (
    FakeAcceptanceModel,
    FakeReportModel,
    FakeResearchModel,
    cited_report,
)

URL = "https://controlled.test/cache"
REQUEST = "How does a local software cache store and reuse data?"
QUESTION = "How does a local software cache store and reuse data?"
REPORT = cited_report(["A cache stores reusable data for later access [S1]."], {"S1": URL})


class SequencedModel:
    model_name = "full-graph-fake"

    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, messages, config=None):
        return AIMessage(content=json.dumps(next(self.responses)))


def _scope_responses(questions=(QUESTION,)):
    return [
        {"allowed": True, "reason": ""},
        {
            "need_clarification": False,
            "question": "",
            "verification": "The cache question is clear; I will prepare a research plan.",
        },
        {
            "research_question": REQUEST,
            "required_answers": ["how a local cache stores and reuses data"],
            "output_requirements": [],
        },
        {
            "questions": list(questions),
            "evidence_needed": ["documentation describing cache storage and reuse"],
            "report_shape": "A short direct answer, then sources.",
        },
    ]


class CountingMCP(ControlledMCP):
    def __init__(self, sources):
        super().__init__(sources)
        self.searches = 0

    def search(self, *args, **kwargs):
        self.searches += 1
        return super().search(*args, **kwargs)


def _graph(stores, scope_model, *, judge_verdicts=None, drafts=(REPORT,), sources=None):
    vfs, index = stores
    mcp = CountingMCP(
        sources or [{"url": URL, "content": "A cache stores reusable data for later access."}]
    )
    researcher = ResearchAgent(ModelRouter(FakeResearchModel()), lambda: mcp)
    writer_model = FakeReportModel(list(drafts))
    nodes = ResearchNodes(
        vfs,
        index,
        researcher,
        report_writer=ReportWriter(ModelRouter(writer_model)),
        report_judge=ReportJudge(
            ModelRouter(
                FakeAcceptanceModel(
                    judge_verdicts
                    or [
                        {
                            "results": [
                                {"criterion": "all", "met": True, "justification": "Answered."}
                            ]
                        }
                    ]
                )
            )
        ),
        faithfulness_evaluator=lambda q, r, c: {"score": 0.96, "unsupported": []},
        require_faithfulness=True,
    )
    graph = build_graph(
        scope_agent=ScopeAgent(ModelRouter(scope_model)),
        vfs=vfs,
        research_nodes=nodes,
        checkpointer=MemorySaver(),
        with_approval=True,
    )
    return graph, mcp, writer_model, nodes


def _interrupt_kind(graph, config):
    snapshot = graph.get_state(config)
    return [i.value["kind"] for task in snapshot.tasks for i in task.interrupts]


def test_no_retrieval_happens_before_initial_plan_approval(stores):
    vfs, _ = stores
    graph, mcp, _, _ = _graph(stores, SequencedModel(_scope_responses()))
    config = {"configurable": {"thread_id": "t"}}

    paused = graph.invoke({"run_id": "r", "thread_id": "t", "request": REQUEST}, config)

    assert _interrupt_kind(graph, config) == ["plan_approval"]
    assert mcp.searches == 0
    assert not (vfs.root / "r" / "research").exists()
    plan = json.loads(vfs.read("r", "planning/plan.v1.json"))
    assert plan == paused["plan"]
    assert plan["original_question"] == REQUEST
    assert plan["required_answers"] == ["how a local cache stores and reuses data"]
    assert plan["tasks"][0]["evidence_needed"] == "documentation describing cache storage and reuse"
    assert plan["report_shape"] == "A short direct answer, then sources."
    assert json.loads(vfs.read("r", "planning/todos.v1.json"))[0]["research_question"] == QUESTION
    interrupt = graph.get_state(config).tasks[0].interrupts[0].value
    assert interrupt["plan"]["plan_hash"] == plan["plan_hash"]


def test_approved_one_task_plan_completes_with_one_worker_and_stops_on_pass(stores):
    graph, mcp, writer, nodes = _graph(stores, SequencedModel(_scope_responses()))
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"run_id": "r", "thread_id": "t", "request": REQUEST}, config)

    result = graph.invoke(Command(resume={"reply": "approve", "actor": "tester"}), config)

    assert _interrupt_kind(graph, config) == ["publication_approval"]
    assert result["stage"] == "Awaiting Approval"
    assert len(result["todos"]) == 1 and result["todos"][0]["status"] == "completed"
    assert mcp.searches == 1
    assert (
        nodes.supervisor.max_concurrent_researchers == 3
    )  # the cap stays; one task uses one worker
    assert result["plan_approval"] == {
        "plan_hash": result["plan"]["plan_hash"],
        "approved": True,
        "outcome": "approve",
        "actor": "tester",
        "kind": "initial",
    }
    assert result["verification_outcome"] == "PASS" and result["draft_version"] == 1
    assert len(writer.prompts("write")) == 1 and not result.get("bvf_history")
    assert "Original user question (verbatim):\n" + REQUEST in writer.prompts("write")[0]


def test_declined_initial_plan_ends_without_research(stores):
    graph, mcp, _, _ = _graph(stores, SequencedModel(_scope_responses()))
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"run_id": "r", "thread_id": "t", "request": REQUEST}, config)

    result = graph.invoke(Command(resume={"approved": False}), config)

    assert result["stage"] == "Cancelled" and result["status"] == "cancelled"
    assert mcp.searches == 0 and not graph.get_state(config).next


def test_plan_change_request_rescopes_and_needs_a_new_approval(stores):
    responses = (
        _scope_responses()
        + _scope_responses()[:1]
        + [
            {
                "need_clarification": False,
                "question": "",
                "verification": "I will cover only official documentation.",
            },
            {
                "research_question": REQUEST + " Use only official documentation.",
                "required_answers": ["how a local cache stores and reuses data"],
                "output_requirements": [],
            },
            {
                "questions": [QUESTION + " (official documentation)"],
                "evidence_needed": ["official docs"],
                "report_shape": "Short answer.",
            },
        ]
    )
    graph, mcp, _, _ = _graph(stores, SequencedModel(responses))
    config = {"configurable": {"thread_id": "t"}}
    first = graph.invoke({"run_id": "r", "thread_id": "t", "request": REQUEST}, config)

    second = graph.invoke(Command(resume={"reply": "Use only official documentation."}), config)

    assert _interrupt_kind(graph, config) == ["plan_approval"]
    assert mcp.searches == 0
    assert second["plan"]["plan_hash"] != first["plan"]["plan_hash"]
    assert second["plan_approval"] is None


def test_research_refuses_to_run_without_approval_of_the_current_plan(stores):
    vfs, index = stores
    nodes = ResearchNodes(
        vfs,
        index,
        ResearchAgent(ModelRouter(FakeResearchModel()), lambda: ControlledMCP([])),
        report_judge=ReportJudge(ModelRouter(FakeAcceptanceModel([{"results": []}]))),
    )
    todo = ResearchTodo(research_question=QUESTION).model_dump(mode="json")
    state = {"run_id": "r", "thread_id": "t", "todos": [todo], "plan": {"plan_hash": "new"}}

    with pytest.raises(ValueError, match="approval of the current plan"):
        nodes.researching({**state, "plan_approval": None})
    with pytest.raises(ValueError, match="approval of the current plan"):
        nodes.researching({**state, "plan_approval": {"approved": True, "plan_hash": "old"}})


def test_replacement_plan_waits_for_approval_after_targeted_research_failed(stores):
    gap = {
        "results": [
            {
                "criterion": "all",
                "met": False,
                "justification": "No figure found.",
                "missing_points": ["the requested figure"],
                "issue_type": "evidence_gap",
            }
        ]
    }
    passed = {"results": [{"criterion": "all", "met": True, "justification": "Answered."}]}
    responses = _scope_responses() + [
        {
            "questions": ["Which official statistics page reports the cache figure?"],
            "evidence_needed": ["an official statistics page"],
            "report_shape": "Short answer.",
        }
    ]
    graph, mcp, _, _ = _graph(
        stores, SequencedModel(responses), judge_verdicts=[gap, gap, passed], drafts=[REPORT]
    )
    config = {"configurable": {"thread_id": "t"}}
    graph.invoke({"run_id": "r", "thread_id": "t", "request": REQUEST}, config)
    paused = graph.invoke(Command(resume={"approved": True}), config)

    assert _interrupt_kind(graph, config) == ["plan_approval"]
    assert paused["plan"]["kind"] == "replacement"
    assert [entry["route"] for entry in paused["bvf_history"]] == ["research", "replan"]
    searches_before = mcp.searches
    assert (
        searches_before == 2
    )  # initial search + one targeted search; nothing for the new task yet

    result = graph.invoke(Command(resume={"approved": True}), config)

    assert result["stage"] == "Awaiting Approval"
    assert mcp.searches == searches_before + 1  # only the newly approved task was researched
    assert len(result["todos"]) == 2 and result["recovery_attempt"] == 2
