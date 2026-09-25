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

    def create_role_router(self, settings, *, mode: str = "live", trace=None, callbacks=None):
        from intelligence_collector.models.role_router import RoleModelRouter
        from intelligence_collector.models.router import ModelRouter

        reasoning = self.create(ModelConfig(
            settings.reasoning_model,
            provider=settings.reasoning_provider,
            reasoning=settings.reasoning_kwargs,
        ))
        reasoning_fallback = None
        if settings.reasoning_fallback_model:
            reasoning_fallback = self.create(ModelConfig(
                settings.reasoning_fallback_model, provider=settings.reasoning_fallback_provider,
            ))
        standard = self.create(ModelConfig(
            settings.standard_model,
            provider=settings.standard_provider,
            reasoning=settings.standard_kwargs,
        ))
        standard_fallback = None
        if settings.standard_fallback_model:
            standard_fallback = self.create(ModelConfig(
                settings.standard_fallback_model, provider=settings.standard_fallback_provider,
            ))
        return RoleModelRouter(
            ModelRouter(reasoning, reasoning_fallback, mode=mode, trace=trace, callbacks=callbacks),
            ModelRouter(standard, standard_fallback, mode=mode, trace=trace, callbacks=callbacks),
        )
