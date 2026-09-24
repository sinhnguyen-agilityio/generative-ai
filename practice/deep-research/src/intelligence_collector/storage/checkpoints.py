from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Any

class Checkpoints:
    def __init__(self, path: str | Path): self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def save(self, thread_id: str, state: dict[str, Any]) -> None:
        data = json.loads(json.dumps(state, default=str)); current = self.load_all(); current[thread_id] = data
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".checkpoint-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(current, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    def load(self, thread_id: str) -> dict[str, Any] | None: return self.load_all().get(thread_id)
    def load_all(self) -> dict[str, Any]:
        if not self.path.exists(): return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
