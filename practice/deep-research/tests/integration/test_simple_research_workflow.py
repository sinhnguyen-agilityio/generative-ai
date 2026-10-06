from langgraph.checkpoint.memory import MemorySaver

from intelligence_collector.agents.report_judge import ReportJudge
from intelligence_collector.agents.report_writer import ReportWriter
from intelligence_collector.contracts import ResearchTodo, TodoStatus
from intelligence_collector.graph.nodes import ResearchNodes
from intelligence_collector.models.router import ModelRouter
from tests.fixtures.fake_models import FakeAcceptanceModel, FakeReportModel
from tests.fixtures.graphs import research_graph


def test_exhausted_core_gap_presents_honest_report_for_approval(stores, tmp_path):
    vfs, index = stores
    draft = "# INCOMPLETE\nThe requested version could not be verified from available evidence.\n\n## Sources\nNone available."
    writer = FakeReportModel([draft])

    class EmptyResearch:
        calls = 0

        def delegate(self, run_id, todos, **kwargs):
            self.calls += 1
            for todo in todos:
                todo.status = TodoStatus.FAILED
            return [(todo, [], []) for todo in todos]

    nodes = ResearchNodes(
        vfs,
        index,
        None,
        report_writer=ReportWriter(ModelRouter(writer)),
        report_judge=ReportJudge(
            ModelRouter(
                FakeAcceptanceModel(
                    [{"results": [{"criterion": "All", "met": True, "justification": "Honest"}]}]
                )
            )
        ),
    )
    empty = EmptyResearch()
    nodes.supervisor = empty
    config = {"configurable": {"thread_id": "r"}}
    saver = MemorySaver()
    graph = research_graph(
        nodes,
        vfs=vfs,
        checkpointer=saver,
        with_approval=True,
        output_dir=str(tmp_path / "exports"),
    )
    result = graph.invoke(
        {
            "run_id": "r",
            "thread_id": "r",
            "scoped_request": {"research_question": "What is the latest stable Python version?"},
            "todos": [
                ResearchTodo(todo_id="t", research_question="Official stable version?").model_dump()
            ],
        },
        config,
    )
    # One pre-draft gap retry, then an honest draft that still requires human report approval.
    assert result["stage"] == "Awaiting Approval"
    assert result.get("__interrupt__")
    assert result["research_gap_attempt"] == 1 and empty.calls == 2
    assert result["verification"]["passed"]
    assert result["approval"] is None and result["publication"] is None
    assert not (tmp_path / "exports" / "r.md").exists()
    # A fresh compiled graph resumes the same paused checkpoint without starting a new run.
    restored = research_graph(nodes, vfs=vfs, checkpointer=saver, with_approval=True)
    assert restored.get_state(config).next == ("approval",)
    assert restored.get_state(config).values["research_gap_attempt"] == 1
    assert len(writer.prompts("write")) == 1
