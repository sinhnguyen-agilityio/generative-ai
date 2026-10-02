"""Application-owned index of durable workflow runs.

LangGraph checkpoints remain authoritative for state. This registry stores only enough metadata
to discover an unfinished thread after process or browser-session loss.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from intelligence_collector.contracts import RunRecord, WorkflowStatus, now


MAX_REQUEST_SUMMARY_LENGTH = 500
TERMINAL_STATUSES = {
    WorkflowStatus.PUBLISHED,
    WorkflowStatus.EXPORTED_INCOMPLETE,
    WorkflowStatus.INCOMPLETE,
    WorkflowStatus.CANCELLED,
}


class RunRegistryConflict(RuntimeError):
    """The requested run/thread identity conflicts with an existing durable record."""


class RunRegistry:
    """SQLite repository for run discovery and lifecycle metadata."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        if self.path.exists() and self.path.is_dir():
            raise RuntimeError(f"Run registry path points to a directory: {self.path}")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS workflow_runs (
                        run_id TEXT PRIMARY KEY,
                        thread_id TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        workflow_status TEXT NOT NULL,
                        current_stage TEXT NOT NULL DEFAULT '',
                        checkpoint_namespace TEXT NOT NULL DEFAULT '',
                        request_summary TEXT NOT NULL DEFAULT '',
                        terminal INTEGER NOT NULL DEFAULT 0 CHECK (terminal IN (0, 1))
                    );
                    CREATE INDEX IF NOT EXISTS idx_workflow_runs_recoverable
                    ON workflow_runs (terminal, updated_at DESC);
                    """
                )
        except (OSError, sqlite3.Error) as exc:
            raise RuntimeError(
                f"Cannot initialize run registry at '{self.path}'. "
                "Check INTELLIGENCE_RUN_REGISTRY_PATH and directory write permissions."
            ) from exc

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _required_identity(run_id: str, thread_id: str) -> tuple[str, str]:
        run_id = str(run_id).strip()
        thread_id = str(thread_id).strip()
        if not run_id or not thread_id:
            raise ValueError("run_id and thread_id must be non-empty")
        return run_id, thread_id

    @staticmethod
    def _summary(value: str) -> str:
        return " ".join(str(value or "").split())[:MAX_REQUEST_SUMMARY_LENGTH]

    @staticmethod
    def _record(row: sqlite3.Row | None) -> RunRecord | None:
        if row is None:
            return None
        return RunRecord.model_validate({**dict(row), "terminal": bool(row["terminal"])})

    def register_run(
        self,
        run_id: str,
        thread_id: str,
        *,
        request_summary: str = "",
        checkpoint_namespace: str = "",
    ) -> RunRecord:
        """Create a run once, or safely refresh metadata for the same identity pair."""
        run_id, thread_id = self._required_identity(run_id, thread_id)
        timestamp = now().isoformat()
        summary = self._summary(request_summary)
        namespace = str(checkpoint_namespace or "").strip()
        try:
            with self._connect() as connection:
                by_run = connection.execute(
                    "SELECT thread_id FROM workflow_runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                by_thread = connection.execute(
                    "SELECT run_id FROM workflow_runs WHERE thread_id = ?", (thread_id,)
                ).fetchone()
                if by_run is not None and by_run["thread_id"] != thread_id:
                    raise RunRegistryConflict(
                        f"run_id '{run_id}' is already bound to another thread"
                    )
                if by_thread is not None and by_thread["run_id"] != run_id:
                    raise RunRegistryConflict(
                        f"thread_id '{thread_id}' is already bound to another run"
                    )
                connection.execute(
                    """
                    INSERT INTO workflow_runs (
                        run_id, thread_id, created_at, updated_at, workflow_status,
                        checkpoint_namespace, request_summary, terminal
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(run_id) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        checkpoint_namespace = CASE
                            WHEN excluded.checkpoint_namespace = ''
                            THEN workflow_runs.checkpoint_namespace
                            ELSE excluded.checkpoint_namespace
                        END,
                        request_summary = CASE
                            WHEN excluded.request_summary = '' THEN workflow_runs.request_summary
                            ELSE excluded.request_summary
                        END
                    """,
                    (
                        run_id,
                        thread_id,
                        timestamp,
                        timestamp,
                        WorkflowStatus.RUNNING.value,
                        namespace,
                        summary,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise RunRegistryConflict("run/thread identity is already registered") from exc
        record = self.get_by_run_id(run_id)
        assert record is not None
        return record

    def update_run(
        self,
        run_id: str,
        thread_id: str,
        *,
        workflow_status: WorkflowStatus,
        current_stage: str = "",
    ) -> RunRecord:
        """Update lifecycle metadata without changing the registered identity."""
        run_id, thread_id = self._required_identity(run_id, thread_id)
        status = WorkflowStatus(workflow_status)
        terminal = status in TERMINAL_STATUSES
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE workflow_runs
                SET updated_at = ?, workflow_status = ?, current_stage = ?, terminal = ?
                WHERE run_id = ? AND thread_id = ?
                """,
                (
                    now().isoformat(),
                    status.value,
                    str(current_stage or ""),
                    int(terminal),
                    run_id,
                    thread_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RunRegistryConflict(
                    f"run '{run_id}' and thread '{thread_id}' are not a registered pair"
                )
        record = self.get_by_run_id(run_id)
        assert record is not None
        return record

    def get_by_run_id(self, run_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM workflow_runs WHERE run_id = ?", (str(run_id),)
            ).fetchone()
        return self._record(row)

    def get_by_thread_id(self, thread_id: str) -> RunRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM workflow_runs WHERE thread_id = ?", (str(thread_id),)
            ).fetchone()
        return self._record(row)

    def list_unfinished(self) -> list[RunRecord]:
        """Return recoverable runs, newest activity first."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM workflow_runs WHERE terminal = 0 ORDER BY updated_at DESC"
            ).fetchall()
        return [self._record(row) for row in rows]
