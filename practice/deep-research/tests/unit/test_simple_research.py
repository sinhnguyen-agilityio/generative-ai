import pytest

from intelligence_collector.contracts import ResearchTodo, TodoStatus
from intelligence_collector.graph.migration import normalize_checkpoint
from intelligence_collector.graph.nodes import ResearchNodes
from intelligence_collector.graph.routing import route_after_research
from intelligence_collector.storage.vfs import VFS


def make_nodes(tmp_path, status="READY"):
    """Readiness comes from the stored evidence: complete (READY), with gaps (PARTIAL) or none
    (INSUFFICIENT)."""
    todo = ResearchTodo(
        todo_id="t", research_question="Which version is stable?", status=TodoStatus.COMPLETED
    )
    nodes = object.__new__(ResearchNodes)
    nodes.vfs = VFS(tmp_path)
    state = {"run_id": "r", "todos": [todo.model_dump(mode="json")], "evidence": []}
    if status != "INSUFFICIENT":
        ref = nodes.vfs.write_json(
            "r",
            "research/t/evidence/result.json",
            {
                "task_id": "t",
                "question": todo.research_question,
                "summary": "Official release details",
                "sources": [{"url": "https://source.test/releases", "excerpt": "Release details"}],
                "sufficient": status == "READY",
                "unresolved_gaps": [] if status == "READY" else ["Main version not verified"],
            },
            "evidence",
            task_id="t",
        )
        state["evidence"] = [ref.model_dump(mode="json")]
    return nodes, state


def test_complete_evidence_is_ready(tmp_path):
    nodes, state = make_nodes(tmp_path)
    result = nodes.research_complete(state)
    assert result["research_status"] == "READY"
    assert not result["result_incomplete"]
    assert route_after_research(result) == "synthesizing"


def test_partial_skips_extra_research(tmp_path):
    nodes, state = make_nodes(tmp_path, "PARTIAL")
    result = nodes.research_complete(state)
    # Gaps reach the writer as limitations; they never mark the result incomplete by themselves.
    assert result["research_status"] == "PARTIAL" and not result["result_incomplete"]
    assert result["research_limitations"][0] == "Main version not verified"
    assert route_after_research(result) == "synthesizing"


def test_insufficient_gets_one_attempt_then_incomplete_synthesis(tmp_path):
    nodes, state = make_nodes(tmp_path, "INSUFFICIENT")
    assessed = {**state, **nodes.research_complete(state)}
    assert route_after_research(assessed) == "prepare_gap_research"
    reopened = {**assessed, **nodes.prepare_gap_research(assessed)}
    exhausted = {**reopened, **nodes.research_complete(reopened)}
    assert exhausted["research_gap_attempt"] == 1
    assert exhausted["research_status"] == "INSUFFICIENT"
    assert route_after_research(exhausted) == "synthesizing"
    with pytest.raises(ValueError, match="exhausted"):
        nodes.prepare_gap_research(exhausted)


def test_legacy_research_judge_failure_goes_to_review():
    state = normalize_checkpoint(
        {"verification_issue": "research_verifier_failure", "research_verifier_attempt": 1}
    )
    assert state["verification_issue"] == "research_failure"
    assert "research_verifier_attempt" not in state
    assert route_after_research(state) == "needs_review"


def test_legacy_budget_is_preserved_not_reset():
    state = normalize_checkpoint(
        {
            "research_gap_attempt": 2,
            "research_gap_budget": 2,
            "progress_assessment": {"ready_for_synthesis": False, "blockers": ["Missing evidence"]},
        }
    )
    assert state["research_gap_attempt"] == 2 and state["research_gap_budget"] == 1
    assert state["result_incomplete"] and state["research_status"] == "INSUFFICIENT"


def test_corrupt_storage_raises_instead_of_inventing_gap(tmp_path):
    nodes, state = make_nodes(tmp_path)
    nodes.vfs.write("r", "research/t/evidence/result.json", "bad json", "evidence")
    with pytest.raises(ValueError):
        nodes.research_complete(state)


@pytest.mark.parametrize("pending", ["replan", "prepare_gap_research"])
def test_legacy_pending_gap_restores_without_a_new_allowance(tmp_path, pending):
    from types import SimpleNamespace
    from intelligence_collector.graph.recovery import WorkflowRecoveryService
    from intelligence_collector.storage.run_registry import RunRegistry

    registry = RunRegistry(tmp_path / "registry.sqlite")
    registry.register_run("r", "r")

    class Graph:
        state = SimpleNamespace(
            values={
                "run_id": "r",
                "thread_id": "r",
                "stage": "Researching",
                "research_gap_attempt": 2,
                "research_gap_budget": 2,
                "todos": [{"todo_id": "t", "research_question": "Version?"}],
            },
            next=(pending,),
            tasks=[],
            metadata={},
        )

        def get_state(self, config):
            return self.state

        def update_state(self, config, state, as_node):
            assert as_node == "research_complete"
            self.state = SimpleNamespace(
                values=state, next=("synthesizing",), tasks=[], metadata={}
            )

        def invoke(self, value, config):
            assert value is None
            assert self.state.values["research_gap_attempt"] == 2
            assert self.state.values["result_incomplete"]
            return {**self.state.values, "stage": "Incomplete"}

    service = WorkflowRecoveryService(Graph(), registry)
    assert service.continue_run("r")["stage"] == "Incomplete"
