from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain.chat_models import init_chat_model


@dataclass(frozen=True)
class ModelConfig:
    model: str
    provider: str | None = None
    timeout: float = 30.0
    reasoning: dict[str, Any] | None = None


class ModelFactory:
    def create(self, config: ModelConfig):
        kwargs = {"timeout": config.timeout, **(config.reasoning or {})}
        if config.provider:
            kwargs["model_provider"] = config.provider
        return init_chat_model(config.model, **kwargs)
