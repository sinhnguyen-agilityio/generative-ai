"""Physical research files exposed through run-scoped virtual paths."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from threading import RLock

_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
_TEMP_PREFIX = ".vfs-"


class VFS:
    """Persist UTF-8 files under ``root/run_id`` without exposing host paths.

    Virtual paths use forward slashes, with an optional leading slash. File writes replace
    the previous version atomically; concurrent writes to the same file are last-write-wins.
    Operations are serialized within an instance, including reads during replacement on Windows.
    This synchronous service can be called through ``asyncio.to_thread`` by future agents.
    The storage directory must be controlled by the application: path validation is not
    an OS sandbox against another process changing links during an operation.
    """

    def __init__(self, root: str | Path = ".data/research") -> None:
        """Create the physical storage root.

        Args:
            root: Application-owned directory containing one workspace per run.
        """
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def _resolve(self, path: str, run_id: str, *, allow_root: bool = False) -> Path:
        """Validate portable virtual paths and reject workspace escapes or linked aliases.

        Args:
            path: Relative or slash-prefixed virtual path.
            run_id: Directory name made of lowercase ASCII letters, digits, hyphens, or underscores.
            allow_root: Whether the workspace directory itself is a valid result.

        Returns:
            Physical path within the requested workspace.

        Raises:
            ValueError: The run ID/path is invalid or traverses a symbolic link or junction.
        """
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", run_id):
            raise ValueError("run_id must be a single lowercase alphanumeric directory name")
        if not path or "\\" in path or any(ord(c) < 32 or c in '<>:"|?*' for c in path):
            raise ValueError("Use a nonempty virtual path with forward slashes")
        if ".." in path.split("/"):
            raise ValueError("Parent traversal is not allowed")
        parts = PurePosixPath(path).parts
        if parts and parts[0] in {"/", "//"}:
            parts = parts[1:]
        for part in (run_id, *parts):
            if part.endswith((".", " ")) or part.split(".")[0].upper() in _RESERVED_NAMES or part.startswith(_TEMP_PREFIX):
                raise ValueError("Reserved or nonportable path component")
        if not parts and not allow_root:
            raise ValueError("A file path is required")
        target = self.root / run_id
        for part in (None, *parts):
            if part is not None:
                target = target / part
            try:
                info = target.lstat()
            except FileNotFoundError:
                continue
            if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                raise ValueError("Symbolic links and junctions are not allowed in a workspace")
        return target

    def write_file(self, path: str, content: str, *, run_id: str) -> str:
        """Atomically create or replace a file, creating parent directories as needed.

        Args:
            path: Virtual destination, such as /topics/topic-001/notes.md.
            content: UTF-8 text; JSON artifacts can be serialized by the caller.
            run_id: Workspace identifier.

        Returns:
            Normalized virtual path for subsequent reads.
        """
        with self._lock:
            target = self._resolve(path, run_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=target.parent, prefix=_TEMP_PREFIX, delete=False) as file:
                    temporary = Path(file.name)
                    file.write(content)
                os.replace(temporary, target)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return "/" + target.relative_to(self.root / run_id).as_posix()

    def read_file(self, path: str, *, run_id: str, offset: int = 0, limit: int | None = 8000) -> str:
        """Read a bounded character window without loading the entire file.

        Args:
            path: Virtual file to read.
            run_id: Workspace identifier.
            offset: Zero-based character offset, not a byte offset.
            limit: Maximum characters returned; None explicitly requests the remainder.

        Returns:
            Text window, or an empty string when the offset is beyond the file.

        Raises:
            ValueError: Offset or limit is negative, or the path is invalid.
            FileNotFoundError: The file does not exist in this workspace.
        """
        if offset < 0 or (limit is not None and limit < 0):
            raise ValueError("offset and limit must be nonnegative")
        with self._lock:
            target = self._resolve(path, run_id)
            with target.open(encoding="utf-8", newline="") as file:
                remaining = offset
                while remaining:
                    skipped = file.read(min(remaining, 8192))
                    if not skipped:
                        return ""
                    remaining -= len(skipped)
                return file.read() if limit is None else file.read(limit)

    def list_files(self, path: str = "/", *, run_id: str) -> list[str]:
        """List virtual file paths recursively beneath a workspace directory.

        Args:
            path: Virtual directory; / lists the entire workspace.
            run_id: Workspace identifier.

        Returns:
            Sorted file paths; an absent directory returns an empty list.

        Raises:
            NotADirectoryError: The requested path is a file.
            ValueError: The directory or a listed entry traverses a linked alias.
        """
        with self._lock:
            directory = self._resolve(path, run_id, allow_root=True)
            if not directory.exists():
                return []
            if not directory.is_dir():
                raise NotADirectoryError(path)
            workspace = self.root / run_id
            files = []
            for entry in directory.iterdir():
                if entry.name.startswith(_TEMP_PREFIX):
                    continue
                virtual = "/" + entry.relative_to(workspace).as_posix()
                entry = self._resolve(virtual, run_id)
                if entry.is_dir():
                    files.extend(self.list_files(virtual, run_id=run_id))
                elif entry.is_file():
                    files.append(virtual)
            return sorted(files)

    def delete_file(self, path: str, *, run_id: str) -> None:
        """Delete one file; an absent file is already considered deleted.

        Args:
            path: Virtual file to delete; directories cannot be deleted through this API.
            run_id: Workspace identifier.
        """
        with self._lock:
            self._resolve(path, run_id).unlink(missing_ok=True)
