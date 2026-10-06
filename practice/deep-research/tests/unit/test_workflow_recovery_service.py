import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from intelligence_collector.graph.build import node_wrapper
from intelligence_collector.graph.recovery import (
    CheckpointIdentityMismatch,
    CheckpointNotFound,
    InvalidRecoveryOperation,
    RecoverableRunNotFound,
    WorkflowRecoveryService,
)
from intelligence_collector.graph.state import GraphState
from intelligence_collector.storage.run_registry import RunRegistry
from tests.fixtures.graphs import approval_graph


def _approval_graph(registry):
    return approval_graph(checkpointer=MemorySaver(), run_registry=registry)


def test_service_lists_and_inspects_an_interrupted_run(tmp_path):
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    graph = _approval_graph(registry)
    config = {"configurable": {"thread_id": "thread-1"}}
    graph.invoke({"run_id": "run-1", "thread_id": "thread-1"}, config)
    service = WorkflowRecoveryService(graph, registry)

    assert [record.run_id for record in service.list_recoverable_runs()] == ["run-1"]
    snapshot = service.inspect("thread-1")
    assert snapshot.run_id == "run-1"
    assert snapshot.has_interrupt is True
    assert snapshot.interrupt_kind == "publication_approval"
    assert snapshot.next_nodes == ["approval"]


def test_service_resumes_approval_on_the_same_thread(tmp_path):
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    graph = _approval_graph(registry)
    config = {"configurable": {"thread_id": "thread-1"}}
    graph.invoke({"run_id": "run-1", "thread_id": "thread-1"}, config)
    service = WorkflowRecoveryService(graph, registry)

    result = service.resume_interrupt("thread-1", {"approved": True, "actor": "reviewer"})

    assert result["stage"] == "Published"
    assert registry.get_by_thread_id("thread-1").terminal is True


def test_service_refuses_wrong_resume_api_and_invalid_approval(tmp_path):
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    graph = _approval_graph(registry)
    config = {"configurable": {"thread_id": "thread-1"}}
    graph.invoke({"run_id": "run-1", "thread_id": "thread-1"}, config)
    service = WorkflowRecoveryService(graph, registry)

    with pytest.raises(InvalidRecoveryOperation, match="use resume_interrupt"):
        service.continue_run("thread-1")
    with pytest.raises(ValidationError, match="rejection requires feedback"):
        service.resume_interrupt("thread-1", {"approved": False})
    with pytest.raises(ValidationError):
        service.resume_interrupt("thread-1", {"approved": True, "node": "publishing"})


def test_service_rejects_missing_checkpoint_and_unknown_thread(tmp_path):
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    graph = _approval_graph(registry)
    registry.register_run("run-1", "thread-1")
    service = WorkflowRecoveryService(graph, registry)

    with pytest.raises(CheckpointNotFound):
        service.inspect("thread-1")
    with pytest.raises(RecoverableRunNotFound):
        service.inspect("unknown")


def test_service_detects_registry_checkpoint_identity_mismatch(tmp_path):
    checkpoint_registry = RunRegistry(tmp_path / "checkpoint-runs.sqlite3")
    graph = _approval_graph(checkpoint_registry)
    config = {"configurable": {"thread_id": "thread-1"}}
    graph.invoke({"run_id": "checkpoint-run", "thread_id": "thread-1"}, config)

    other_registry = RunRegistry(tmp_path / "other-runs.sqlite3")
    other_registry.register_run("different-run", "thread-1")
    service = WorkflowRecoveryService(graph, other_registry)

    with pytest.raises(CheckpointIdentityMismatch):
        service.inspect("thread-1")


def test_service_continues_a_failed_node_without_restarting_the_run(tmp_path):
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    attempts = []

    def flaky(state):
        attempts.append("called")
        if len(attempts) == 1:
            raise RuntimeError("transient failure")
        return {**state, "stage": "Recovered"}

    builder = StateGraph(GraphState)
    builder.add_node("work", node_wrapper("work", flaky, run_registry=registry))
    builder.add_edge(START, "work")
    builder.add_edge("work", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "thread-1"}}

    with pytest.raises(RuntimeError, match="transient failure"):
        graph.invoke({"run_id": "run-1", "thread_id": "thread-1"}, config)

    service = WorkflowRecoveryService(graph, registry)
    before = service.inspect("thread-1")
    assert before.has_interrupt is False
    assert before.next_nodes == ["work"]

    result = service.continue_run("thread-1")

    assert result["stage"] == "Recovered"
    assert attempts == ["called", "called"]
