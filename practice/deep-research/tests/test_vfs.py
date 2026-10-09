"""Physical VFS persistence, workspace isolation, path checks, and atomic writes."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from storage.vfs import VFS


def test_files_are_physical_persistent_and_isolated_between_runs(tmp_path):
    """Persist identical virtual paths independently and reopen them through a new instance."""
    vfs = VFS(tmp_path)
    assert vfs.write_file("topics/t1/notes.md", "Evidence A", run_id="run-a") == "/topics/t1/notes.md"
    vfs.write_file("/topics/t1/notes.md", "Evidence B", run_id="run-b")
    assert (tmp_path / "run-a/topics/t1/notes.md").read_text(encoding="utf-8") == "Evidence A"
    reopened = VFS(tmp_path)
    assert reopened.read_file("/topics/t1/notes.md", run_id="run-a") == "Evidence A"
    assert reopened.read_file("/topics/t1/notes.md", run_id="run-b") == "Evidence B"
    with pytest.raises(FileNotFoundError):
        reopened.read_file("/topics/t1/notes.md", run_id="run-c")


def test_read_windows_count_unicode_characters_and_preserve_newlines(tmp_path):
    """Use character offsets so multibyte Unicode and CRLF text round-trip correctly."""
    vfs = VFS(tmp_path)
    content = "Nghiên cứu 🔎\r\n" + "x" * 9000
    vfs.write_file("notes.txt", content, run_id="run")
    assert vfs.read_file("notes.txt", run_id="run") == content[:8000]
    assert vfs.read_file("notes.txt", run_id="run", offset=7, limit=10) == content[7:17]
    assert vfs.read_file("notes.txt", run_id="run", offset=8500, limit=None) == content[8500:]
    assert vfs.read_file("notes.txt", run_id="run", offset=len(content) + 1) == ""
    assert vfs.read_file("notes.txt", run_id="run", limit=0) == ""
    with pytest.raises(ValueError):
        vfs.read_file("notes.txt", run_id="run", offset=-1)
    with pytest.raises(ValueError):
        vfs.read_file("notes.txt", run_id="run", limit=-1)


@pytest.mark.parametrize("path", ["", "/", "../secret", "/topics/../../secret", r"C:\secret", r"..\secret", "notes:stream", "CON.txt", "notes.", "notes ", "bad\x00name", "/.vfs-hidden"])
def test_invalid_file_paths_cannot_write_outside_the_workspace(tmp_path, path):
    """Reject traversal, Windows aliases, control characters, and internal temporary names."""
    with pytest.raises(ValueError):
        VFS(tmp_path).write_file(path, "data", run_id="run")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("run_id", ["", "..", "../other", "run/other", r"run\other", "con", "run:other", "RUN"])
def test_invalid_run_ids_are_rejected(tmp_path, run_id):
    """Prevent run identifiers from selecting another workspace or reserved host path."""
    with pytest.raises(ValueError):
        VFS(tmp_path).write_file("notes.md", "data", run_id=run_id)


def test_list_and_delete_only_affect_the_requested_workspace(tmp_path):
    """List nested files deterministically and delete without affecting another run."""
    vfs = VFS(tmp_path)
    vfs.write_file("/notes.md", "a", run_id="run-a")
    vfs.write_file("/topics/t1/report.json", "{}", run_id="run-a")
    vfs.write_file("/notes.md", "b", run_id="run-b")
    assert vfs.list_files(run_id="run-a") == ["/notes.md", "/topics/t1/report.json"]
    assert vfs.list_files("/topics", run_id="run-a") == ["/topics/t1/report.json"]
    assert vfs.list_files(run_id="absent") == []
    with pytest.raises(NotADirectoryError):
        vfs.list_files("/notes.md", run_id="run-a")
    vfs.delete_file("/notes.md", run_id="run-a")
    vfs.delete_file("/notes.md", run_id="run-a")
    assert vfs.read_file("/notes.md", run_id="run-b") == "b"


def test_failed_replacement_preserves_previous_file_and_removes_temporary_file(tmp_path, monkeypatch):
    """Keep the original artifact intact when final publication of a write fails."""
    vfs = VFS(tmp_path)
    vfs.write_file("/notes.md", "original", run_id="run")

    def fail_replace(*args):
        """Simulate a filesystem failure before replacement."""
        raise OSError("replacement failed")

    monkeypatch.setattr("storage.vfs.os.replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        vfs.write_file("/notes.md", "replacement", run_id="run")
    assert vfs.read_file("/notes.md", run_id="run") == "original"
    assert [p.name for p in (tmp_path / "run").iterdir()] == ["notes.md"]


def test_concurrent_writes_produce_one_complete_file(tmp_path):
    """Concurrent writers must never concatenate or partially overwrite artifacts."""
    vfs = VFS(tmp_path)
    contents = [str(i) * 10000 for i in range(8)]

    def write(content):
        """Write one candidate artifact to the shared destination."""
        vfs.write_file("/notes.md", content, run_id="run")

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, contents))
    assert vfs.read_file("/notes.md", run_id="run", limit=None) in contents
    assert [p.name for p in (tmp_path / "run").iterdir()] == ["notes.md"]


def test_linked_workspace_cannot_alias_another_run(tmp_path):
    """Reject linked paths even when their destination is inside the storage root."""
    vfs = VFS(tmp_path)
    vfs.write_file("/notes.md", "private", run_id="other")
    try:
        (tmp_path / "run").symlink_to(tmp_path / "other", target_is_directory=True)
    except OSError:
        pytest.skip("Creating symbolic links is unavailable on this host")
    with pytest.raises(ValueError, match="links"):
        vfs.read_file("/notes.md", run_id="run")
    with pytest.raises(ValueError, match="links"):
        vfs.write_file("/notes.md", "changed", run_id="run")
    assert vfs.read_file("/notes.md", run_id="other") == "private"
