import sqlite3

import pytest

from intelligence_collector.runtime import open_checkpoint_connection as _open_checkpoint_connection


def test_checkpoint_connection_creates_missing_parent_directory(tmp_path):
    target = tmp_path / "nested" / "runtime" / "scope.sqlite3"

    connection = _open_checkpoint_connection(str(target))
    try:
        connection.execute("CREATE TABLE probe (value TEXT)")
        connection.commit()
    finally:
        connection.close()

    assert target.is_file()


def test_checkpoint_connection_rejects_directory_path(tmp_path):
    with pytest.raises(RuntimeError, match="points to a directory"):
        _open_checkpoint_connection(str(tmp_path))


def test_checkpoint_connection_reports_actionable_sqlite_error(monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr("intelligence_collector.runtime.sqlite3.connect", fail)
    target = tmp_path / "scope.sqlite3"

    with pytest.raises(RuntimeError, match="INTELLIGENCE_CHECKPOINT_PATH"):
        _open_checkpoint_connection(str(target))
