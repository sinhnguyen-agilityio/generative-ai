import json

from langchain_core.messages import AIMessage

from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.agents.report_judge import ReportJudge
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.nodes import ResearchNodes
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP
from intelligence_collector.agents.report_writer import ReportWriter
from tests.fixtures.fake_models import FakeAcceptanceModel, FakeReportModel, FakeResearchModel, cited_report


class SequencedModel:
    model_name = "full-graph-fake"

    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, messages, config=None):
        return AIMessage(content=json.dumps(next(self.responses)))


def test_full_graph_executes_guarded_research_to_approval_ready(stores):
    vfs, index = stores
    model = SequencedModel([
        {"allowed": True, "reason": ""},
        {
            "need_clarification": False,
            "question": "",
            "verification": "The cache research scope is clear; research will begin.",
        },
        {
            "research_question": (
                "I want a technical report explaining local software cache behavior using "
                "technical sources and citing the retrieved evidence."
            )
        },
        {
            "questions": ["How does a local software cache store and reuse data?"],
        },
    ])
    mcp = ControlledMCP([{
        "url": "https://controlled.test/cache",
        "content": "A cache stores reusable data for later access.",
    }])
    researcher = ResearchAgent(ModelRouter(FakeResearchModel()), lambda: mcp)
    writer = ReportWriter(ModelRouter(FakeReportModel([cited_report(
        ["A cache stores reusable data for later access [S1]."],
        {"S1": "https://controlled.test/cache"},
    )])))
    judge = ReportJudge(ModelRouter(FakeAcceptanceModel([{
        "results": [{"criterion": "all criteria", "met": True, "justification": "Satisfied."}],
    }])))
    graph = build_graph(
        scope_agent=ScopeAgent(ModelRouter(model)),
        vfs=vfs,
        research_nodes=ResearchNodes(vfs, index, researcher, report_writer=writer, report_judge=judge),
    )

    result = graph.invoke({
        "run_id": "full-graph",
        "thread_id": "full-thread",
        "request": "Explain cache behavior for local software using technical sources.",
    })

    assert result["stage"] == "Awaiting Approval"
    assert result["verification"]["passed"] is True
    assert result["verification_history"] == [result["verification"]]
    assert result["todos"][0]["status"] == "completed"
    assert result["evidence"] and result["index_refs"]
    assert vfs.read("full-graph", "planning/todos.v1.json")
    assert mcp.requests[0]["query"] == "How does a local software cache store and reuse data?"
    assert vfs.read("full-graph", "supervisor/progress/latest.json")
    assert vfs.read("full-graph", "reports/latest.md")
