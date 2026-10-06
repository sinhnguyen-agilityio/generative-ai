"""End to end through the graph: a verified report, then the reviewer asks to add a section."""

import json

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from intelligence_collector.agents.extension_judge import ExtensionJudge
from intelligence_collector.agents.report_judge import ReportJudge
from intelligence_collector.agents.report_writer import ReportWriter
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.graph.nodes import ResearchNodes
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.fake_models import (
    FakeAcceptanceModel,
    FakeReportModel,
    FakeResearchModel,
    cited_report,
)
from tests.integration.test_approval_gate import GOOD, PASS, URL, _initial_state, _router
from tests.fixtures.graphs import research_graph

URL2 = "https://controlled.test/eviction"
REQUEST = "Add a section on eviction specifically."
TOPIC = "How does a local cache evict entries?"
EXTENDED = cited_report(
    [
        "A cache stores reusable data for later access [S1].",
        "Old entries are evicted when the cache is full [S2].",
    ],
    {"S1": URL, "S2": URL2},
)


class ScriptedModel:
    model_name = "fake-extension-judge"

    def __init__(self, payload):
        self.payload, self.calls = payload, []

    def invoke(self, messages, config=None):
        self.calls.append(list(messages))
        return AIMessage(content=json.dumps(self.payload))


def _graph(stores, tmp_path, judge_payload, report_model):
    vfs, index = stores
    searches = []

    def mcp():
        # First run finds the cache page; any later research finds the eviction page.
        rows = (
            [{"url": URL, "content": "A cache stores reusable data for later access."}]
            if not searches
            else [{"url": URL2, "content": "Old entries are evicted when the cache is full."}]
        )
        searches.append(1)
        return ControlledMCP(rows)

    nodes = ResearchNodes(
        vfs,
        index,
        ResearchAgent(ModelRouter(FakeResearchModel()), mcp),
        report_writer=ReportWriter(ModelRouter(report_model)),
        report_judge=ReportJudge(ModelRouter(FakeAcceptanceModel([PASS]))),
        recovery_budget=3,
    )
    judge_model = ScriptedModel(judge_payload)
    _, router = _router({"route": "extend", "reason": "Adds a section.", "topic": TOPIC})
    graph = research_graph(
        nodes,
        vfs=vfs,
        checkpointer=MemorySaver(),
        with_approval=True,
        output_dir=str(tmp_path / "exports"),
        feedback_router=router,
        extension_judge=ExtensionJudge(ModelRouter(judge_model)),
    )
    return graph, searches, judge_model


def test_stored_evidence_is_enough_so_the_section_is_written_without_research(stores, tmp_path):
    with_section = GOOD.replace(
        "## Sources", "## Storage\nA cache keeps data for reuse [S1].\n\n## Sources"
    )
    report_model = FakeReportModel([GOOD, with_section])
    graph, searches, judge_model = _graph(
        stores,
        tmp_path,
        {
            "sufficient": True,
            "relevant_source_ids": ["S1"],
            "missing_points": [],
            "summary": "A cache stores reusable data for later access.",
        },
        report_model,
    )
    config = {"configurable": {"thread_id": "approval-run"}}

    graph.invoke(_initial_state(), config)
    paused = graph.invoke(Command(resume={"reply": REQUEST}), config)

    assert paused["stage"] == "Awaiting Approval"
    assert paused["feedback_route"] == "extend"
    assert paused["extension"]["sufficient"] is True
    assert len(searches) == 1  # no new research
    assert judge_model.calls
    assert paused["draft"] == with_section
    write_prompts = report_model.prompts("write")
    assert REQUEST in write_prompts[1] and "Current report to revise" in write_prompts[1]


def test_missing_evidence_researches_only_the_added_task(stores, tmp_path):
    report_model = FakeReportModel([GOOD, EXTENDED])
    graph, searches, _ = _graph(
        stores,
        tmp_path,
        {
            "sufficient": False,
            "relevant_source_ids": [],
            "missing_points": ["eviction policy when the cache is full"],
            "summary": "",
        },
        report_model,
    )
    config = {"configurable": {"thread_id": "approval-run"}}

    graph.invoke(_initial_state(), config)
    paused = graph.invoke(Command(resume={"reply": REQUEST}), config)

    assert paused["stage"] == "Awaiting Approval"
    assert paused["extension"]["sufficient"] is False
    assert len(searches) == 2  # one sub-agent for the original task, one for the added task
    assert [todo["status"] for todo in paused["todos"]] == ["completed", "completed"]
    assert paused["todos"][1]["research_question"] == TOPIC
    assert paused["draft"] == EXTENDED
    assert paused["verification"]["passed"] is True
