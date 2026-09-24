from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path
from typing import Any

from intelligence_collector.contracts import ArtifactRef


class VFS:
    def __init__(self, root: str | Path): self.root = Path(root).resolve()
    def _safe(self, run_id: str, relative: str) -> Path:
        base = (self.root / run_id).resolve(); target = (base / relative).resolve()
        if target != base and base not in target.parents: raise ValueError("cross-run VFS path")
        return target
    def write(self, run_id: str, relative: str, content: str | bytes, kind: str, *, task_id: str | None = None) -> ArtifactRef:
        data = content.encode() if isinstance(content, str) else content
        target = self._safe(run_id, relative); target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp): os.unlink(tmp)
        return ArtifactRef(run_id=run_id, kind=kind, path=str(target), content_hash=hashlib.sha256(data).hexdigest(), task_id=task_id)
    def write_json(self, run_id: str, relative: str, value: Any, kind: str, **kwargs: Any) -> ArtifactRef:
        return self.write(run_id, relative, json.dumps(value, default=str, indent=2), kind, **kwargs)
    def read(self, run_id: str, relative: str) -> bytes: return self._safe(run_id, relative).read_bytes()

