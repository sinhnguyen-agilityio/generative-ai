import pytest

from intelligence_collector.contracts import ResearchTodo, TodoStatus
from intelligence_collector.agents.worker import _evidence_path, already_researched
from intelligence_collector.graph.nodes import Node, ResearchNodes
from intelligence_collector.graph.routing import route_after_research, route_after_verify
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.source_verification import load_catalog
from tests.fixtures.graphs import research_graph


def test_gap_preparation_reopens_only_affected_todo_and_preserves_old_attempt(tmp_path):
    vfs = VFS(tmp_path)
    first = ResearchTodo(todo_id="a", research_question="Question A", status=TodoStatus.COMPLETED)
    other = ResearchTodo(todo_id="b", research_question="Question B", status=TodoStatus.COMPLETED)
    vfs.write_json("run", _evidence_path(first), {"summary": "old"}, "evidence_bundle")
    nodes = object.__new__(ResearchNodes)
    nodes.vfs = vfs
    result = nodes.prepare_gap_research(
        {
            "run_id": "run",
            "todos": [first.model_dump(), other.model_dump()],
            "research_gaps": [{"todo_id": "a", "instruction": "Find the missing number"}],
            "approval": {"approved": True},
        }
    )
    reopened = ResearchTodo.model_validate(result["todos"][0])
    assert reopened.todo_id == "a"
    assert reopened.research_attempt == 2
    assert reopened.status is TodoStatus.PLANNED
    assert result["todos"][1]["status"] == "completed"
    assert not already_researched(vfs, "run", reopened)
    assert already_researched(vfs, "run", first)
    assert result["approval"] is None
    assert result["research_gap_attempt"] == 1


def test_latest_attempt_resolves_gap_without_changing_source_citations(tmp_path):
    vfs = VFS(tmp_path)
    refs = []
    for attempt, sufficient in [(1, False), (2, True)]:
        todo = ResearchTodo(todo_id="a", research_question="Question A", research_attempt=attempt)
        refs.append(
            vfs.write_json(
                "run",
                _evidence_path(todo),
                {
                    "task_id": "a",
                    "question": "Question A",
                    "summary": f"Attempt {attempt}",
                    "sources": [
                        {"locator": f"https://example.test/{attempt}", "excerpt": "Evidence"}
                    ],
                    "sufficient": sufficient,
                    "unresolved_gaps": [] if sufficient else ["Missing data"],
                },
                "evidence_bundle",
                task_id="a",
            ).model_dump()
        )
    catalog = load_catalog(vfs, "run", refs)
    assert len(catalog.bundles) == 1
    assert catalog.bundles[0].summary == "Attempt 2"
    assert not catalog.gaps
    assert catalog.sources["S1"].url.endswith("/1")
    assert catalog.sources["S2"].url.endswith("/2")


@pytest.mark.parametrize(
    "issue,expected",
    [
        ("draft_defect", "fix"),
        ("evidence_gap", "fix"),
        ("plan_gap", "fix"),
        ("verifier_failure", Node.RETRY_VERIFIER),
    ],
)
def test_report_routes_failures_to_the_appropriate_loop(issue, expected):
    assert route_after_verify({"verification_issue": issue}) == expected


def test_research_and_verifier_budgets_are_independent():
    assert (
        route_after_verify(
            {"verification_issue": "verifier_failure", "verifier_attempt": 2, "recovery_attempt": 0}
        )
        == "needs_review"
    )
    assert (
        route_after_research({"research_gap_attempt": 2, "research_gap_budget": 2})
        == Node.SYNTHESIZE
    )
    assert (
        route_after_research({"progress_assessment": {"ready_for_synthesis": True}})
        == Node.SYNTHESIZE
    )


def test_optional_gaps_allow_partial_synthesis(tmp_path):
    nodes = object.__new__(ResearchNodes)
    nodes.vfs = VFS(tmp_path)
    todo = ResearchTodo(todo_id="a", research_question="Question A", status=TodoStatus.COMPLETED)
    nodes.vfs.write_json(
        "run",
        _evidence_path(todo),
        {
            "summary": "Best available answer",
            "sources": [{"locator": "https://example.test"}],
            "sufficient": False,
            "unresolved_gaps": ["Missing latency measurement"],
        },
        "evidence_bundle",
    )
    result = nodes.research_complete({"run_id": "run", "todos": [todo.model_dump()]})
    assert result["progress_assessment"]["ready_for_synthesis"]
    assert result["research_status"] == "PARTIAL"
    assert route_after_research(result) == Node.SYNTHESIZE


def test_graph_repairs_evidence_gap_before_first_synthesis(stores):
    from intelligence_collector.agents.report_judge import ReportJudge
    from intelligence_collector.agents.report_writer import ReportWriter
    from intelligence_collector.agents.researcher import ResearchAgent
    from intelligence_collector.models.router import ModelRouter
    from intelligence_collector.tools.mcp import ControlledMCP
    from tests.fixtures.fake_models import FakeResearchModel, FakeReportModel, FakeAcceptanceModel

    vfs, index = stores
    searches = []

    def mcp():
        # The first research finds nothing, so readiness asks for one gap-research round.
        searches.append(1)
        rows = [{"url": "https://example.test/data", "content": "Measured research evidence"}]
        return ControlledMCP(rows if len(searches) > 1 else [])

    researcher = ResearchAgent(ModelRouter(FakeResearchModel()), mcp)
    writer = FakeReportModel(
        ["# Report\nEvidence [S1]\n\n## Sources\n- [S1] https://example.test/data"]
    )
    nodes = ResearchNodes(
        vfs,
        index,
        researcher,
        report_writer=ReportWriter(ModelRouter(writer)),
        report_judge=ReportJudge(
            ModelRouter(
                FakeAcceptanceModel(
                    [{"results": [{"criterion": "All", "met": True, "justification": "Supported"}]}]
                )
            )
        ),
    )
    result = research_graph(nodes).invoke(
        {
            "run_id": "run",
            "thread_id": "thread",
            "scoped_request": {
                "research_question": "Compare measurements with primary research evidence"
            },
            "todos": [ResearchTodo(todo_id="a", research_question="Measurements?").model_dump()],
        }
    )
    assert len(searches) == 2
    assert len(writer.prompts("write")) == 1
    assert result["todos"][0]["research_attempt"] == 2
    assert result["research_gap_attempt"] == 1
    assert result["verification"]["passed"]
