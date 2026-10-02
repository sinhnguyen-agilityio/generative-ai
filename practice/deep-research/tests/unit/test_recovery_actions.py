import pytest

from intelligence_collector.graph.nodes import RecoveryNodes
from intelligence_collector.graph.routing import route_after_recovery
from intelligence_collector.storage.vfs import VFS


def state(action, **kwargs):
    return {"run_id": "run", "thread_id": "thread", "draft": "Unverified draft",
            "verification": {"passed": False, "repair_details": "Unsupported claim"},
            "last_error": "Unsupported claim", "approval": {"approved": True},
            "recovery_attempt": 3, "stall_count": 2,
            "recovery_decision": {"action": action, **kwargs}}


@pytest.mark.parametrize("action", ["retry_report", "revise"])
def test_repair_invalidates_verification_and_starts_bounded_allowance(tmp_path, action):
    payload = {"additional_attempts": 2} if action == "retry_report" else {"feedback": "Use bullets"}
    result = RecoveryNodes(VFS(tmp_path)).apply(state(action, **payload))
    assert result["verification"] is None
    assert result["approval"] is None
    assert result["recovery_attempt"] == 0
    assert result["stall_count"] == 0
    assert 1 <= result["recovery_budget"] <= 2
    assert result["reviewer_requests"]
    assert result["recovery_decision"] is None
    assert route_after_recovery(result) == action


def test_gap_research_only_plans_the_authorized_question(tmp_path):
    result = RecoveryNodes(VFS(tmp_path)).apply(state("research", feedback="Find latency evidence"))
    assert len(result["todos"]) == 1
    assert result["todos"][0]["status"] == "planned"
    assert result["todos"][0]["research_question"] == "Find latency evidence"


def test_incomplete_export_is_labeled_and_never_published(tmp_path):
    vfs = VFS(tmp_path)
    result = RecoveryNodes(vfs).apply(state("export_incomplete"))
    assert result["status"] == "exported_incomplete"
    assert result["publication"] is None
    assert "INCOMPLETE" in vfs.read("run", "reports/incomplete.md").decode()


def test_cancel_and_rescope(tmp_path):
    nodes = RecoveryNodes(VFS(tmp_path), can_rescope=True)
    assert nodes.apply(state("cancel"))["status"] == "cancelled"
    result = nodes.apply(state("rescope", feedback="Compare only SQLite"))
    assert result["request"] == "Compare only SQLite"
    assert result["draft"] == ""
    with pytest.raises(ValueError, match="Scope Agent"):
        RecoveryNodes(VFS(tmp_path)).apply(state("rescope", feedback="Change scope"))


def test_recovery_checkpoint_survives_restart_and_cancel_is_terminal(tmp_path):
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import START, END, StateGraph
    from intelligence_collector.graph.nodes import await_recovery
    from intelligence_collector.graph.state import GraphState
    from intelligence_collector.graph.instrumentation import node_wrapper
    from intelligence_collector.graph.recovery import WorkflowRecoveryService
    from intelligence_collector.storage.run_registry import RunRegistry

    registry = RunRegistry(tmp_path / "runs.sqlite3")
    nodes = RecoveryNodes(VFS(tmp_path / "vfs"))

    def build(connection):
        builder = StateGraph(GraphState)
        builder.add_node("await_recovery", node_wrapper(
            "await_recovery", await_recovery, run_registry=registry))
        builder.add_node("apply", node_wrapper("apply", nodes.apply, run_registry=registry))
        builder.add_edge(START, "await_recovery")
        builder.add_edge("await_recovery", "apply")
        builder.add_edge("apply", END)
        return builder.compile(checkpointer=SqliteSaver(connection))

    path = tmp_path / "checkpoints.sqlite3"
    config = {"configurable": {"thread_id": "thread"}}
    with sqlite3.connect(path, check_same_thread=False) as connection:
        paused = build(connection).invoke({**state("cancel"), "stage": "NEEDS_REVIEW"}, config)
        assert paused.get("__interrupt__")
    with sqlite3.connect(path, check_same_thread=False) as connection:
        service = WorkflowRecoveryService(build(connection), registry)
        result = service.resume_interrupt("thread", {"action": "cancel"})
        assert result["status"] == "cancelled"
        assert registry.get_by_thread_id("thread").terminal
        assert not service.list_recoverable_runs()
