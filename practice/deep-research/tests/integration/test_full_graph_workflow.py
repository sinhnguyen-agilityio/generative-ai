import json

from langchain_core.messages import AIMessage

from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.agents.worker import ControlledMCP
from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.nodes import ResearchNodes
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.storage.vfs import VFS


class SequencedModel:
    model_name = "full-graph-fake"

    def __init__(self, responses):
        self.responses = iter(responses)

    def invoke(self, messages, config=None):
        return AIMessage(content=json.dumps(next(self.responses)))


def test_full_graph_executes_guarded_research_to_approval_ready(stores):
    vfs, index = stores
    model = SequencedModel([
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
        {"questions": ["How does a local software cache store and reuse data?"]},
    ])
    mcp = ControlledMCP([{
        "url": "https://controlled.test/cache",
        "content": "A cache stores reusable data for later access.",
        "sufficient": True,
    }])
    graph = build_graph(
        scope_agent=ScopeAgent(ModelRouter(model, mode="fake")),
        vfs=vfs,
        research_nodes=ResearchNodes(vfs, index, mcp),
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
    assert vfs.read("full-graph", "supervisor/progress/latest.json")
    assert vfs.read("full-graph", "reports/latest.md")
