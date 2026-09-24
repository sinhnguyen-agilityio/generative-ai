from dataclasses import dataclass
import json
from typing import TypeVar

from pydantic import BaseModel
from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class FallbackEvent:
    role: str
    reason: str

T = TypeVar("T", bound=BaseModel)


class ModelRouter:
    def __init__(self, primary=None, fallback=None, *, max_attempts: int = 2, on_fallback=None, mode: str = "live", trace=None):
        self.primary, self.fallback = primary, fallback
        self.max_attempts = max_attempts
        self.on_fallback = on_fallback
        self.events: list[FallbackEvent] = []
        self.mode = mode
        self.trace = trace

    def invoke(self, role, prompt):
        if self.mode == "live" and self.primary is None:
            raise RuntimeError("live model router requires a configured primary model")
        if self.mode == "fake" and self.primary is None:
            raise RuntimeError("fake model router requires an explicit fake model")
        try:
            return self.primary(role, prompt)
        except Exception as exc:
            if self.max_attempts < 2:
                raise
            event = FallbackEvent(role=role, reason=str(exc))
            self.events.append(event)
            if self.on_fallback:
                self.on_fallback(event)
            return self.fallback(role, prompt)

    def invoke_structured(self, role: str, prompt: str, schema: type[T], *, prompt_version: str, run_id: str = "", thread_id: str = "") -> T:
        raw = self.invoke(role, prompt)
        if isinstance(raw, schema):
            result = raw
        elif isinstance(raw, dict):
            result = schema.model_validate(raw)
        elif isinstance(raw, str):
            result = schema.model_validate(json.loads(raw))
        else:
            raise TypeError(f"model output for {role} is not structured")
        if self.trace:
            self.trace(role=role, prompt_version=prompt_version, run_id=run_id, thread_id=thread_id, decision=result.model_dump(mode="json"))
        return result

    def invoke_messages(self, role: str, messages: list[BaseMessage], schema: type[T], *, prompt_version: str, run_id: str = "", thread_id: str = "") -> T:
        raw = self.invoke(role, messages)
        if hasattr(raw, "content"):
            raw = raw.content
        if isinstance(raw, schema):
            result = raw
        elif isinstance(raw, dict):
            result = schema.model_validate(raw)
        elif isinstance(raw, str):
            result = schema.model_validate(json.loads(raw))
        else:
            raise TypeError(f"model output for {role} is not structured")
        if self.trace:
            self.trace(role=role, prompt_version=prompt_version, run_id=run_id, thread_id=thread_id, decision=result.model_dump(mode="json"))
        return result
