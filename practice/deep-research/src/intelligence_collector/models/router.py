from dataclasses import dataclass
import json
import re
from typing import TypeVar

from pydantic import BaseModel
from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class FallbackEvent:
    role: str
    reason: str
    selected_model: str = ""

T = TypeVar("T", bound=BaseModel)
MAX_MODEL_ATTEMPTS = 3
SECRET_MARKERS = ("api_key", "apikey", "secret", "password", "token", "authorization")


def _redact_text(value: object) -> str:
    text = str(value)
    text = re.sub(
        r"(?i)\b(api[_-]?key|secret|password|token|authorization)\s*[:=]\s*([^\s,;]+)",
        r"\1=[REDACTED]",
        text,
    )
    return re.sub(r"(?i)\bbearer\s+[a-z0-9._-]{8,}", "Bearer [REDACTED]", text)


def _redact_value(value):
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if any(marker in key.lower() for marker in SECRET_MARKERS)
            else _redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return _redact_text(value) if isinstance(value, str) else value


class ModelRouter:
    def __init__(self, primary=None, fallback=None, *, max_attempts: int = 2, on_fallback=None, mode: str = "live", trace=None, callbacks=None):
        if not isinstance(max_attempts, int) or not 1 <= max_attempts <= MAX_MODEL_ATTEMPTS:
            raise ValueError(f"max_attempts must be between 1 and {MAX_MODEL_ATTEMPTS}")
        self.primary, self.fallback = primary, fallback
        self.max_attempts = max_attempts
        self.on_fallback = on_fallback
        self.events: list[FallbackEvent] = []
        self.mode = mode
        self.trace = trace
        self.callbacks = list(callbacks or [])
        self.last_model = ""
        self.fallback_used = False

    @staticmethod
    def _model_name(model) -> str:
        if model is None:
            return ""
        return str(getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__)

    @staticmethod
    def _invoke_model(model, role: str, prompt, config=None):
        if hasattr(model, "invoke"):
            return model.invoke(prompt, config=config) if config else model.invoke(prompt)
        if callable(model):
            return model(role, prompt)
        raise TypeError("configured model must be callable or expose invoke()")

    def _trace(self, **event):
        if not self.trace:
            return
        try:
            self.trace(**event)
        except TypeError:
            # Preserve compatibility with simple test tracers while keeping rich metadata
            # for the production tracing adapter.
            self.trace(event)

    def _parse(self, role: str, raw, schema: type[T]) -> T:
        if hasattr(raw, "content"):
            raw = raw.content
        if isinstance(raw, schema):
            return raw
        if isinstance(raw, dict):
            return schema.model_validate(raw)
        if isinstance(raw, str):
            return schema.model_validate(json.loads(raw))
        raise TypeError(f"model output for {role} is not structured")

    @staticmethod
    def _prompt_text(prompt) -> str:
        """Render the exact model input as readable text for trace inspection."""
        if isinstance(prompt, list):
            blocks = []
            for message in prompt:
                role = getattr(message, "type", type(message).__name__).upper()
                content = _redact_text(getattr(message, "content", str(message)))
                blocks.append(f"[{role}]\n{content}")
            return "\n\n".join(blocks)
        return _redact_text(prompt)

    @staticmethod
    def _output_text(raw) -> str:
        value = getattr(raw, "content", raw)
        if isinstance(value, BaseModel):
            return _redact_text(value.model_dump_json(indent=2))
        if isinstance(value, (dict, list)):
            return json.dumps(_redact_value(value), ensure_ascii=False, indent=2, default=str)
        return _redact_text(value)

    def invoke(self, role, prompt, *, run_id="", thread_id="", prompt_version=""):
        if self.mode == "live" and self.primary is None:
            raise RuntimeError("live model router requires a configured primary model")
        if self.mode == "fake" and self.primary is None:
            raise RuntimeError("fake model router requires an explicit fake model")
        self.fallback_used = False
        config = None
        if self.callbacks:
            config = {
                "callbacks": self.callbacks,
                "run_name": f"model.{role}",
                "metadata": {
                    "langfuse_session_id": thread_id,
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "prompt_version": prompt_version,
                    "role": role,
                },
            }
        try:
            self.last_model = self._model_name(self.primary)
            return self._invoke_model(self.primary, role, prompt, config)
        except Exception as exc:
            if self.max_attempts < 2:
                raise
            if self.fallback is None:
                raise RuntimeError(f"primary model failed for {role} and no fallback is configured") from exc
            self.fallback_used = True
            self.last_model = self._model_name(self.fallback)
            event = FallbackEvent(
                role=role,
                reason=_redact_text(exc),
                selected_model=self.last_model,
            )
            self.events.append(event)
            if self.on_fallback:
                self.on_fallback(event)
            return self._invoke_model(self.fallback, role, prompt, config)

    def invoke_structured(self, role: str, prompt: str, schema: type[T], *, prompt_version: str, run_id: str = "", thread_id: str = "") -> T:
        return self._invoke_and_parse(role, prompt, schema, prompt_version, run_id, thread_id)

    def invoke_messages(self, role: str, messages: list[BaseMessage], schema: type[T], *, prompt_version: str, run_id: str = "", thread_id: str = "") -> T:
        return self._invoke_and_parse(role, messages, schema, prompt_version, run_id, thread_id, len(messages))

    def _invoke_and_parse(self, role, prompt, schema, prompt_version, run_id, thread_id, message_count=None):
        errors = []
        attempts = max(1, self.max_attempts)
        prompt_text = self._prompt_text(prompt)
        for attempt in range(1, attempts + 1):
            try:
                raw = self.invoke(
                    role,
                    prompt,
                    run_id=run_id,
                    thread_id=thread_id,
                    prompt_version=prompt_version,
                )
                result = self._parse(role, raw, schema)
                event = {
                    "role": role,
                    "prompt_version": prompt_version,
                    "run_id": run_id,
                    "thread_id": thread_id,
                    "decision": _redact_value(result.model_dump(mode="json")),
                    "selected_model": self.last_model,
                    "fallback_used": self.fallback_used,
                    "attempt": attempt,
                    "input_text": prompt_text,
                    "output_text": _redact_text(result.model_dump_json(indent=2)),
                }
                if message_count is not None:
                    event["message_count"] = message_count
                self._trace(**event)
                return result
            except Exception as exc:
                errors.append(_redact_text(exc))
                self._trace(
                    role=role,
                    prompt_version=prompt_version,
                    run_id=run_id,
                    thread_id=thread_id,
                    selected_model=self.last_model,
                    fallback_used=self.fallback_used,
                    attempt=attempt,
                    input_text=prompt_text,
                    output_text=f"{type(exc).__name__}: invalid structured output",
                    validation_error=f"{type(exc).__name__}: invalid structured output",
                )
        raise ValueError(f"invalid structured output for {role} after {attempts} attempts: {'; '.join(errors)}")
