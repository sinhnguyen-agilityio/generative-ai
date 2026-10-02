import pytest

from intelligence_collector.contracts import Stage, WorkflowStatus
from intelligence_collector.graph.build import node_wrapper
from intelligence_collector.storage.run_registry import RunRegistry


def test_node_wrapper_registers_and_updates_waiting_run(tmp_path):
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    wrapped = node_wrapper(
        "clarify_with_user",
        lambda state: {
            **state,
            "stage": Stage.SCOPING.value,
            "scope_assessment": {"need_clarification": True},
        },
        run_registry=registry,
    )

    wrapped({"run_id": "run", "thread_id": "thread", "request": "Compare databases"})

    record = registry.get_by_run_id("run")
    assert record is not None
    assert record.workflow_status is WorkflowStatus.WAITING_FOR_CLARIFICATION
    assert record.current_stage == Stage.SCOPING.value


def test_node_wrapper_records_failure_as_recoverable(tmp_path):
    registry = RunRegistry(tmp_path / "runs.sqlite3")

    def fail(_state):
        raise RuntimeError("failed")

    wrapped = node_wrapper("researching", fail, run_registry=registry)
    with pytest.raises(RuntimeError, match="failed"):
        wrapped({"run_id": "run", "thread_id": "thread", "stage": Stage.RESEARCHING.value})

    record = registry.get_by_run_id("run")
    assert record is not None
    assert record.workflow_status is WorkflowStatus.FAILED
    assert record.terminal is False
