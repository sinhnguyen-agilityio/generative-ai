import hashlib

from intelligence_collector.contracts import FeedbackDecision, VerificationResult
from intelligence_collector.graph.nodes import ApprovalNodes
from intelligence_collector.source_verification import (
    catalog_from_bundles,
    citation_errors,
    load_catalog,
)
from intelligence_collector.storage.vfs import VFS


def test_citations_require_known_ids_and_matching_source_urls():
    catalog = catalog_from_bundles(
        [{"task_id": "t", "sources": [{"url": "https://source.test/a"}]}]
    )
    good = "Fact [S1].\n## Sources\n- [S1] https://source.test/a"
    assert citation_errors(good, catalog) == []
    assert citation_errors(good.replace("source.test", "attacker.test"), catalog)
    assert citation_errors(good.replace("S1", "S7"), catalog)
    assert citation_errors("Fact [S1]", catalog)


def test_active_catalog_excludes_old_scope_without_renumbering(tmp_path):
    vfs = VFS(tmp_path)
    refs = []
    for todo in ["old", "new"]:
        ref = vfs.write_json(
            "r",
            f"{todo}.json",
            {"task_id": todo, "sources": [{"url": f"https://source.test/{todo}"}]},
            "evidence",
            task_id=todo,
        )
        refs.append(ref.model_dump(mode="json"))
    catalog = load_catalog(vfs, "r", refs, active_todo_ids=["new"])
    assert list(catalog.sources) == ["S2"]
    assert [bundle.todo_id for bundle in catalog.bundles] == ["new"]


def test_conditional_approval_cannot_be_overridden_by_classifier(tmp_path):
    class Router:
        def route(self, *args, **kwargs):
            return FeedbackDecision(route="approve")

    nodes = ApprovalNodes(VFS(tmp_path), str(tmp_path / "exports"), Router())
    result = nodes.classify_feedback(
        {"human_feedback": "Approve but fix the citations", "draft": "draft"}
    )
    assert result["feedback_route"] == "revise"
    assert result["approval"] is None


def test_failed_publication_invalidates_approval_and_verification(tmp_path):
    nodes = ApprovalNodes(VFS(tmp_path), str(tmp_path / "exports"))
    verification = VerificationResult(artifact_hash=hashlib.sha256(b"old").hexdigest(), passed=True)
    state = {
        "run_id": "r",
        "draft": "changed",
        "verification": verification.model_dump(mode="json"),
    }
    state["approval"] = nodes._approval_for(state, True)
    result = nodes.publishing(state)
    assert result["stage"] == "NEEDS_REVIEW"
    assert result["verification"] is None and result["approval"] is None
    assert not result["publication"]["published"]


def test_legacy_latest_artifacts_retain_immutable_history(tmp_path):
    vfs = VFS(tmp_path)
    vfs.write("r", "planning/todos.v1.json", "old", "todos")
    vfs.write("r", "planning/todos.v1.json", "new", "todos")
    archive = hashlib.sha256(b"old").hexdigest()
    assert vfs.read("r", f"planning/todos.{archive}.json") == b"old"
    assert vfs.read("r", "planning/todos.v1.json") == b"new"


def test_index_failure_resume_reuses_durable_retrieved_outcome(tmp_path):
    import pytest
    from intelligence_collector.agents.researcher import ResearchOutcome
    from intelligence_collector.agents.supervisor import ResearchSupervisor
    from intelligence_collector.contracts import ResearchFindings, ResearchTodo

    class Researcher:
        calls = 0

        def research(self, question, **kwargs):
            self.calls += 1
            return ResearchOutcome(
                question=question,
                search_budget=1,
                mcp_server="test",
                mcp_tool="search",
                sources=[{"url": "https://source.test/a", "content": "facts"}],
                findings=ResearchFindings(summary="facts", sufficient=True),
            )

    class Index:
        fail = True

        def index(self, chunks):
            if self.fail:
                raise RuntimeError("index unavailable")

    from intelligence_collector.models.router import ModelRouter
    from tests.fixtures.fake_models import FakeResearchModel

    researcher, index = Researcher(), Index()
    supervisor = ResearchSupervisor(
        researcher, VFS(tmp_path), index, router=ModelRouter(FakeResearchModel())
    )
    todo = ResearchTodo(todo_id="t", research_question="Question?")
    with pytest.raises(RuntimeError, match="persistence failed"):
        supervisor.delegate("r", [todo])
    index.fail = False
    assert supervisor.delegate("r", [todo])[0][1]
    assert researcher.calls == 1


def test_authentication_errors_are_not_retried_as_invalid_schema():
    import pytest
    from intelligence_collector.contracts import FeedbackDecision
    from intelligence_collector.models.router import ModelRouter

    class Unauthorized(Exception):
        status_code = 401

    class Model:
        calls = 0

        def invoke(self, messages):
            self.calls += 1
            raise Unauthorized("invalid credential")

    model = Model()
    with pytest.raises(RuntimeError, match="model call"):
        ModelRouter(model, max_attempts=3).invoke_messages("feedback_router", [], FeedbackDecision)
    assert model.calls == 1
