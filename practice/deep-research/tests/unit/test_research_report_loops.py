import pytest

from intelligence_collector.contracts import ResearchTodo, TodoStatus
from intelligence_collector.agents.worker import _evidence_path, already_researched
from intelligence_collector.graph.nodes import Node, ResearchNodes
from intelligence_collector.graph.routing import route_after_research, route_after_verify
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.source_verification import load_catalog


def test_gap_preparation_reopens_only_affected_todo_and_preserves_old_attempt(tmp_path):
    vfs = VFS(tmp_path)
    first = ResearchTodo(todo_id="a", research_question="Question A", status=TodoStatus.COMPLETED)
    other = ResearchTodo(todo_id="b", research_question="Question B", status=TodoStatus.COMPLETED)
    vfs.write_json("run", _evidence_path(first), {"summary": "old"}, "evidence_bundle")
    nodes = object.__new__(ResearchNodes)
    nodes.vfs = vfs
    result = nodes.prepare_gap_research({
        "run_id": "run", "todos": [first.model_dump(), other.model_dump()],
        "research_gaps": [{"todo_id": "a", "instruction": "Find the missing number"}],
        "approval": {"approved": True},
    })
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
        refs.append(vfs.write_json("run", _evidence_path(todo), {
            "task_id": "a", "question": "Question A", "summary": f"Attempt {attempt}",
            "sources": [{"locator": f"https://example.test/{attempt}", "excerpt": "Evidence"}],
            "sufficient": sufficient, "unresolved_gaps": [] if sufficient else ["Missing data"],
        }, "evidence_bundle", task_id="a").model_dump())
    catalog = load_catalog(vfs, "run", refs)
    assert len(catalog.bundles) == 1
    assert catalog.bundles[0].summary == "Attempt 2"
    assert not catalog.gaps
    assert catalog.sources["S1"].url.endswith("/1")
    assert catalog.sources["S2"].url.endswith("/2")


@pytest.mark.parametrize("issue,expected", [
    ("draft_defect", "fix"), ("evidence_gap", Node.PREPARE_GAP_RESEARCH),
    ("plan_gap", Node.REPLAN), ("verifier_failure", Node.RETRY_VERIFIER),
])
def test_report_routes_failures_to_the_appropriate_loop(issue, expected):
    assert route_after_verify({"verification_issue": issue}) == expected


def test_research_and_verifier_budgets_are_independent():
    assert route_after_verify({"verification_issue": "verifier_failure", "verifier_attempt": 2,
                               "recovery_attempt": 0}) == "needs_review"
    assert route_after_research({"research_gap_attempt": 2, "research_gap_budget": 2}) == Node.NEEDS_REVIEW
    assert route_after_research({"progress_assessment": {"ready_for_synthesis": True}}) == Node.SYNTHESIZE


def test_missing_evidence_blocks_synthesis_before_report_is_written(tmp_path):
    from intelligence_collector.contracts import ProgressAssessment
    class Supervisor:
        def assess(self, *args):
            return ProgressAssessment(ready_for_synthesis=True)
    nodes = object.__new__(ResearchNodes)
    nodes.vfs = VFS(tmp_path)
    nodes.supervisor = Supervisor()
    nodes.research_assessor = None
    todo = ResearchTodo(todo_id="a", research_question="Question A", status=TodoStatus.COMPLETED)
    nodes.vfs.write_json("run", _evidence_path(todo), {
        "summary": "Best available answer", "sources": [{"locator": "https://example.test"}],
        "sufficient": False, "unresolved_gaps": ["Missing latency measurement"],
    }, "evidence_bundle")
    result = nodes.research_complete({"run_id": "run", "todos": [todo.model_dump()]})
    assert not result["progress_assessment"]["ready_for_synthesis"]
    assert result["research_gaps"][0]["todo_id"] == "a"
    assert route_after_research(result) == Node.PREPARE_GAP_RESEARCH


def test_semantic_research_judge_rejects_omitted_questions():
    from intelligence_collector.agents.research_judge import ResearchEvidenceJudge
    from intelligence_collector.contracts import ResearchEvidenceVerdict
    class Router:
        def invoke_messages(self, *args, **kwargs):
            return ResearchEvidenceVerdict(questions=[
                {"todo_id": "a", "sufficient": True}], plan_complete=True)
    todos = [ResearchTodo(todo_id=ident, research_question="Question " + ident) for ident in ["a", "b"]]
    with pytest.raises(ValueError, match="every todo exactly once"):
        ResearchEvidenceJudge(Router()).assess("Brief", todos, "Evidence")


def test_graph_repairs_evidence_gap_before_first_synthesis(stores):
    from intelligence_collector.agents.report_judge import ReportJudge
    from intelligence_collector.agents.report_writer import ReportWriter
    from intelligence_collector.agents.researcher import ResearchAgent
    from intelligence_collector.contracts import ResearchEvidenceVerdict
    from intelligence_collector.graph.build import build_graph
    from intelligence_collector.models.router import ModelRouter
    from intelligence_collector.tools.mcp import ControlledMCP
    from tests.fixtures.fake_models import FakeResearchModel, FakeReportModel, FakeAcceptanceModel

    class Assessor:
        calls = 0
        def assess(self, brief, todos, evidence, **kwargs):
            self.calls += 1
            return ResearchEvidenceVerdict(questions=[{
                "todo_id": "a", "sufficient": self.calls > 1,
                "missing_evidence": "Find the missing measurement" if self.calls == 1 else "",
            }], plan_complete=True)

    vfs, index = stores
    researcher = ResearchAgent(ModelRouter(FakeResearchModel()), lambda: ControlledMCP([
        {"url": "https://example.test/data", "content": "Measured research evidence"}]))
    writer = FakeReportModel(["# Report\nEvidence [S1]"])
    assessor = Assessor()
    nodes = ResearchNodes(vfs, index, researcher, research_assessor=assessor,
                          report_writer=ReportWriter(ModelRouter(writer)),
                          report_judge=ReportJudge(ModelRouter(FakeAcceptanceModel([
                              {"results": [{"criterion": "All", "met": True,
                                            "justification": "Supported"}]}]))))
    result = build_graph(research_nodes=nodes).invoke({
        "run_id": "run", "thread_id": "thread",
        "scoped_request": {"research_question": "Compare measurements with primary research evidence"},
        "todos": [ResearchTodo(todo_id="a", research_question="Measurements?").model_dump()],
    })
    assert assessor.calls == 2
    assert len(writer.prompts("write")) == 1
    assert result["todos"][0]["research_attempt"] == 2
    assert result["research_gap_attempt"] == 1
    assert result["verification"]["passed"]
