"""Builds the chat models and the two `ModelRouter` tiers the agents are wired with."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from langchain.chat_models import init_chat_model

from intelligence_collector.models.router import MAX_MODEL_ATTEMPTS, ModelRouter
from intelligence_collector.models.token_limits import output_limits_from_env
from intelligence_collector.config import _bounded_int


@dataclass(frozen=True)
class ModelConfig:
    """How to build one chat model: name, provider, timeout and extra provider kwargs."""

    model: str
    provider: str | None = None
    timeout: float = 30.0
    # Extra provider kwargs passed straight to `init_chat_model` (e.g. {"reasoning_effort": "high"}).
    kwargs: dict[str, Any] | None = None


class ModelRouters(NamedTuple):
    """The two tiers; each agent is constructed with the one it needs.

    reasoning: scope, planner, report judge (verify), feedback classifier.
    standard: researcher, compressor, synthesizer, fixer, scope guardrail.
    """

    reasoning: ModelRouter
    standard: ModelRouter


class ModelFactory:
    """Creates chat models through ``init_chat_model`` and builds the reasoning and standard
    ``ModelRouter`` tiers.
    """

    def create(self, config: ModelConfig):
        """Instantiate one chat model through LangChain's provider-agnostic `init_chat_model`."""
        kwargs = {"timeout": config.timeout, "max_retries": 0, **(config.kwargs or {})}
        if config.provider:
            kwargs["model_provider"] = config.provider
        return init_chat_model(config.model, **kwargs)

    def create_routers(self, settings, *, trace=None, callbacks=None) -> ModelRouters:
        """Build the reasoning and standard tiers from `RoleModelSettings`, each with its own
        optional fallback model. `trace` and `callbacks` are shared by both tiers."""

        limits = output_limits_from_env()
        attempts = _bounded_int("INTELLIGENCE_MODEL_MAX_ATTEMPTS", 1, MAX_MODEL_ATTEMPTS)

        def tier(model, provider, kwargs, fallback_model, fallback_provider, timeout, **router_kwargs) -> ModelRouter:
            """Build one tier: the primary model plus an optional fallback created without the extra
            kwargs.
            """
            primary = self.create(ModelConfig(model, provider=provider, timeout=timeout, kwargs=kwargs))
            # The fallback runs with default settings: it only has to be a reliable safety net.
            fallback = (
                self.create(ModelConfig(fallback_model, provider=fallback_provider, timeout=timeout))
                if fallback_model
                else None
            )
            return ModelRouter(primary, fallback, trace=trace, callbacks=callbacks,
                               output_limits=limits, **router_kwargs)

        return ModelRouters(
            # Retries are explicit and shared by both tiers; graph repair allowances
            # remain separate. A configured fallback still runs once per attempt.
            reasoning=tier(
                settings.reasoning_model, settings.reasoning_provider, settings.reasoning_kwargs,
                settings.reasoning_fallback_model, settings.reasoning_fallback_provider,
                settings.reasoning_timeout,
                max_attempts=attempts,
            ),
            standard=tier(
                settings.standard_model, settings.standard_provider, settings.standard_kwargs,
                settings.standard_fallback_model, settings.standard_fallback_provider,
                settings.standard_timeout,
                max_attempts=attempts,
            ),
        )
