from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from langchain_core.messages import messages_from_dict, messages_to_dict


CHECKPOINT_DEFAULTS: dict[str, Any] = {
    "scoped_request": None,
    "todos": [],
    "todo_ids": [],
    "artifacts": [],
    "evidence": [],
    "index_refs": [],
    "progress_assessment": None,
    "draft": "",
    "previous_draft": "",
    "verification": None,
    "verification_history": [],
    "recovery_attempt": 0,
    "recovery_budget": 3,
    "progress_fingerprint": "",
    "approval": None,
    "publication": None,
}


def _json_value(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class Checkpoints:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _serialize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        payload = {**CHECKPOINT_DEFAULTS, **state}
        messages = payload.pop("messages", None)
        serialized = _json_value(payload)
        if messages is not None:
            serialized["messages"] = messages_to_dict(messages)
        return json.loads(json.dumps(serialized, default=str))

    @staticmethod
    def _deserialize_state(state: dict[str, Any]) -> dict[str, Any]:
        payload = dict(state)
        messages = payload.get("messages")
        if isinstance(messages, list) and all(
            isinstance(message, dict) and "type" in message and "data" in message
            for message in messages
        ):
            payload["messages"] = messages_from_dict(messages)
        return payload

    def save(self, thread_id: str, state: dict[str, Any]) -> None:
        if not thread_id.strip():
            raise ValueError("checkpoint save requires a non-empty thread_id")
        data = self._serialize_state(state)
        current = self.load_all()
        current[thread_id] = data
        fd, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=".checkpoint-")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(current, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def load(self, thread_id: str) -> dict[str, Any] | None:
        payload = self.load_all().get(thread_id)
        return self._deserialize_state(payload) if payload is not None else None

    def save_state(self, state) -> None:
        payload = state.model_dump(mode="json") if hasattr(state, "model_dump") else state
        thread_id = state.thread_id if hasattr(state, "thread_id") else payload.get("thread_id", "")
        self.save(thread_id, payload)

    def load_state(self, thread_id: str, model=None):
        payload = self.load(thread_id)
        return model.model_validate(payload) if payload and model else payload

    def load_all(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}
