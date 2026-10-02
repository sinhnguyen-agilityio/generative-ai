import sqlite3

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from intelligence_collector.graph.instrumentation import node_wrapper
from intelligence_collector.graph.recovery import WorkflowRecoveryService
from intelligence_collector.graph.state import GraphState
from intelligence_collector.storage.run_registry import RunRegistry


@pytest.mark.parametrize("kind,decision", [
    ("clarification_required", {"reply": "Use the latest stable release"}),
    ("publication_approval", {"approved": True}),
    ("recovery_required", {"action": "cancel"}),
])
def test_interrupt_survives_connection_restart_on_same_thread(tmp_path, kind, decision):
    checkpoint_path = tmp_path / "checkpoints.sqlite"
    registry_path = tmp_path / "runs.sqlite"
    calls = []

    def build(connection, registry):
        builder = StateGraph(GraphState)

        def work(state):
            calls.append("work")
            return {**state, "draft": "Already completed work"}

        def wait(state):
            response = interrupt({"kind": kind})
            return {**state, "current_action": str(response), "stage": "Recovered"}

        builder.add_node("work", node_wrapper("work", work, run_registry=registry))
        builder.add_node("wait", node_wrapper("wait", wait, run_registry=registry))
        builder.add_edge(START, "work")
        builder.add_edge("work", "wait")
        builder.add_edge("wait", END)
        return builder.compile(checkpointer=SqliteSaver(connection))

    with sqlite3.connect(checkpoint_path, check_same_thread=False) as connection:
        graph = build(connection, RunRegistry(registry_path))
        graph.invoke({"run_id": "run", "thread_id": "thread"},
                     {"configurable": {"thread_id": "thread"}})
    connection.close()
    with sqlite3.connect(checkpoint_path, check_same_thread=False) as connection:
        service = WorkflowRecoveryService(build(connection, RunRegistry(registry_path)),
                                          RunRegistry(registry_path))
        assert service.load_state("thread")["draft"] == "Already completed work"
        assert service.inspect("thread").interrupt_kind == kind
        result = service.resume_interrupt("thread", decision)
        assert result["run_id"] == "run" and result["thread_id"] == "thread"
        assert result["stage"] == "Recovered"
        assert calls == ["work"]
    connection.close()


def test_duplicate_publication_preserves_file_and_rechecks_approval(tmp_path):
    import hashlib

    from intelligence_collector.contracts import VerificationResult
    from intelligence_collector.publication import approval_for, publish

    content = "Approved exact report"
    digest = hashlib.sha256(content.encode()).hexdigest()
    verification = VerificationResult(artifact_hash=digest, passed=True)
    approval = approval_for(digest, verification.verification_id, True, "reviewer")
    target = tmp_path / "report.md"
    assert publish(content, str(target), approval, verification).published
    modified = target.stat().st_mtime_ns
    assert publish(content, str(target), approval, verification).published
    assert target.stat().st_mtime_ns == modified
    approval.approved = False
    assert not publish(content, str(target), approval, verification).published


def test_legacy_ended_review_is_discovered_and_bridged_to_interrupt(tmp_path):
    from intelligence_collector.graph.nodes import await_recovery

    path = tmp_path / "legacy.sqlite"
    registry = RunRegistry(tmp_path / "registry.sqlite")
    config = {"configurable": {"thread_id": "thread"}}
    with sqlite3.connect(path, check_same_thread=False) as connection:
        old = StateGraph(GraphState)
        old.add_node("needs_review", lambda state: {**state, "stage": "NEEDS_REVIEW"})
        old.add_edge(START, "needs_review")
        old.add_edge("needs_review", END)
        old.compile(checkpointer=SqliteSaver(connection)).invoke(
            {"run_id": "run", "thread_id": "thread"}, config)
    connection.close()
    with sqlite3.connect(path, check_same_thread=False) as connection:
        new = StateGraph(GraphState)
        new.add_node("needs_review", lambda state: state)
        new.add_node("await_recovery", await_recovery)
        new.add_edge(START, "needs_review")
        new.add_edge("needs_review", "await_recovery")
        new.add_edge("await_recovery", END)
        service = WorkflowRecoveryService(new.compile(checkpointer=SqliteSaver(connection)), registry)
        assert service.discover_legacy_runs()[0].run_id == "run"
        restored = service.load_state("thread")
        assert restored["run_id"] == "run"
        assert service.inspect("thread").interrupt_kind == "recovery_required"
    connection.close()
