from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from intelligence_collector.contracts import ArtifactRef
from intelligence_collector.storage.vfs import VFS


class ArtifactStore:
    """Run-scoped artifact catalog backed by the path-safe VFS."""

    def __init__(self, vfs: VFS):
        self.vfs = vfs
        self._versions: dict[tuple[str, str], int] = defaultdict(int)
        self._catalog: dict[str, list[ArtifactRef]] = defaultdict(list)

    def write(self, run_id: str, kind: str, relative_path: str, value: Any, *, task_id: str | None = None) -> ArtifactRef:
        key = (run_id, kind)
        self._versions[key] += 1
        payload = value if isinstance(value, str) else json.dumps(
            value, default=str, indent=2)
        ref = self.vfs.write(run_id, relative_path,
                             payload, kind, task_id=task_id)
        versioned = ref.model_copy(update={"version": self._versions[key]})
        self._catalog[run_id].append(versioned)
        return versioned

    def list(self, run_id: str, kind: str | None = None) -> list[ArtifactRef]:
        refs = self._catalog.get(run_id, [])
        return [ref for ref in refs if kind is None or ref.kind == kind]
