"""Run-scoped virtual file system: every artifact of a run lives under ``<root>/<run_id>/``, written
atomically, with path escapes rejected.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from intelligence_collector.contracts import ArtifactRef


SECRET_KEYS = (
    "api_key",
    "apikey",
    "secret",
    "authorization",
    "credential",
    "password",
    "token",
)


def _artifact_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if any(marker in key.lower() for marker in SECRET_KEYS)
            else _artifact_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_artifact_safe(item) for item in value]
    return value


class VFS:
    """Path-safe, atomic file storage per run."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _safe(self, run_id: str, relative: str) -> Path:
        if not run_id.strip() or Path(run_id).is_absolute():
            raise ValueError("invalid VFS run_id")
        base = (self.root / run_id).resolve()
        if base == self.root or self.root not in base.parents:
            raise ValueError("cross-run VFS path")
        relative_path = Path(relative)
        if not relative.strip() or relative_path.is_absolute():
            raise ValueError("invalid VFS relative path")
        target = (base / relative_path).resolve()
        if target == base or base not in target.parents:
            raise ValueError("cross-run VFS path")
        return target

    def write(
        self,
        run_id: str,
        relative: str,
        content: str | bytes,
        kind: str,
        *,
        task_id: str | None = None,
    ) -> ArtifactRef:
        """Atomically write `content` under the run and return its ``ArtifactRef`` (with the content
        hash).
        """
        data = content.encode() if isinstance(content, str) else content
        target = self._safe(run_id, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Retain immutable revisions while keeping legacy .v1 paths as latest aliases.
        if ".v1." in target.name and target.exists():
            previous = target.read_bytes()
            digest = hashlib.sha256(previous).hexdigest()
            archive = target.with_name(target.name.replace(".v1.", f".{digest}."))
            if not archive.exists():
                self.write(run_id, str(archive.relative_to(self.root / run_id)), previous, kind,
                           task_id=task_id)
        fd, temporary = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return ArtifactRef(
            run_id=run_id,
            kind=kind,
            path=str(target),
            content_hash=hashlib.sha256(data).hexdigest(),
            task_id=task_id,
        )

    def write_json(
        self,
        run_id: str,
        relative: str,
        value: Any,
        kind: str,
        **kwargs: Any,
    ) -> ArtifactRef:
        """Write `value` as indented JSON, redacting credential-shaped keys."""
        payload = json.dumps(_artifact_safe(value), default=str, indent=2)
        return self.write(run_id, relative, payload, kind, **kwargs)

    def read(self, run_id: str, relative: str) -> bytes:
        """Read one file of the run as bytes."""
        return self._safe(run_id, relative).read_bytes()
