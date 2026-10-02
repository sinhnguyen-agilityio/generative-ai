import pytest

from intelligence_collector.contracts import WorkflowStatus
from intelligence_collector.storage.run_registry import (
    MAX_REQUEST_SUMMARY_LENGTH,
    RunRegistry,
    RunRegistryConflict,
)


def test_registry_registers_and_finds_a_run(tmp_path):
    registry = RunRegistry(tmp_path / "nested" / "runs.sqlite3")

    created = registry.register_run(
        "run-1",
        "thread-1",
        request_summary="  Compare   PostgreSQL and SQLite  ",
        checkpoint_namespace="research",
    )

    assert created.run_id == "run-1"
    assert created.thread_id == "thread-1"
    assert created.workflow_status is WorkflowStatus.RUNNING
    assert created.request_summary == "Compare PostgreSQL and SQLite"
    assert created.checkpoint_namespace == "research"
    assert registry.get_by_thread_id("thread-1") == created


def test_register_is_idempotent_but_identity_pairs_are_immutable(tmp_path):
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    first = registry.register_run("run-1", "thread-1", request_summary="Initial")
    second = registry.register_run("run-1", "thread-1", request_summary="Updated")

    assert first.created_at == second.created_at
    assert second.updated_at >= first.updated_at
    assert second.request_summary == "Updated"

    with pytest.raises(RunRegistryConflict, match="another thread"):
        registry.register_run("run-1", "thread-2")
    with pytest.raises(RunRegistryConflict, match="another run"):
        registry.register_run("run-2", "thread-1")


def test_registry_lists_only_unfinished_runs_newest_first(tmp_path):
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    first = registry.register_run("run-1", "thread-1")
    registry.register_run("run-2", "thread-2")
    registry.update_run(
        "run-1",
        "thread-1",
        workflow_status=WorkflowStatus.WAITING_FOR_APPROVAL,
        current_stage="Awaiting Approval",
    )
    registry.update_run(
        "run-2",
        "thread-2",
        workflow_status=WorkflowStatus.PUBLISHED,
        current_stage="Published",
    )

    unfinished = registry.list_unfinished()

    assert [record.run_id for record in unfinished] == ["run-1"]
    assert unfinished[0].terminal is False
    assert unfinished[0].updated_at >= first.updated_at


def test_failed_run_remains_discoverable_for_later_recovery(tmp_path):
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    registry.register_run("run-1", "thread-1")
    failed = registry.update_run(
        "run-1",
        "thread-1",
        workflow_status=WorkflowStatus.FAILED,
        current_stage="Researching",
    )

    assert failed.terminal is False
    assert registry.list_unfinished() == [failed]


def test_registry_rejects_unregistered_identity_and_bounds_summary(tmp_path):
    registry = RunRegistry(tmp_path / "runs.sqlite3")
    with pytest.raises(RunRegistryConflict, match="not a registered pair"):
        registry.update_run(
            "missing",
            "missing",
            workflow_status=WorkflowStatus.RUNNING,
        )

    record = registry.register_run(
        "run-1", "thread-1", request_summary="x" * (MAX_REQUEST_SUMMARY_LENGTH + 20)
    )
    assert len(record.request_summary) == MAX_REQUEST_SUMMARY_LENGTH
