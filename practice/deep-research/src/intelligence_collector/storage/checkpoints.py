from __future__ import annotations
import json
from pathlib import Path
from typing import Any

class Checkpoints:
    def __init__(self, path: str | Path): self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    def save(self, thread_id: str, state: dict[str, Any]) -> None:
        data = json.loads(json.dumps(state, default=str)); current = self.load_all(); current[thread_id] = data
        self.path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    def load(self, thread_id: str) -> dict[str, Any] | None: return self.load_all().get(thread_id)
    def load_all(self) -> dict[str, Any]:
        if not self.path.exists(): return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

