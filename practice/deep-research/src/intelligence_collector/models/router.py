from dataclasses import dataclass


@dataclass(frozen=True)
class FallbackEvent:
    role: str
    reason: str


class ModelRouter:
    def __init__(self, primary, fallback, *, max_attempts: int = 2, on_fallback=None):
        self.primary, self.fallback = primary, fallback
        self.max_attempts = max_attempts
        self.on_fallback = on_fallback
        self.events: list[FallbackEvent] = []

    def invoke(self, role, prompt):
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
