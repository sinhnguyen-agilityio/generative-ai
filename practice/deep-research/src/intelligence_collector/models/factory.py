"""Builds the chat models and the two `ModelRouter` tiers the agents are wired with."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

from langchain.chat_models import init_chat_model

from intelligence_collector.models.router import MAX_MODEL_ATTEMPTS, ModelRouter


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
        kwargs = {"timeout": config.timeout, **(config.kwargs or {})}
        if config.provider:
            kwargs["model_provider"] = config.provider
        return init_chat_model(config.model, **kwargs)

    def create_routers(self, settings, *, trace=None, callbacks=None) -> ModelRouters:
        """Build the reasoning and standard tiers from `RoleModelSettings`, each with its own
        optional fallback model. `trace` and `callbacks` are shared by both tiers."""

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
            return ModelRouter(primary, fallback, trace=trace, callbacks=callbacks, **router_kwargs)

        return ModelRouters(
            # Groq's own strict-structured-output decoding for gpt-oss is intermittently flaky
            # (json_validate_failed on a fraction of otherwise-identical requests -- a known,
            # unresolved Groq-side issue, not something a schema/prompt change can fix). The
            # reasoning tier carries verify's report-judge call, so give it every retry the
            # router allows before it ever has to fall back.
            reasoning=tier(
                settings.reasoning_model, settings.reasoning_provider, settings.reasoning_kwargs,
                settings.reasoning_fallback_model, settings.reasoning_fallback_provider,
                settings.reasoning_timeout,
                max_attempts=MAX_MODEL_ATTEMPTS,
            ),
            standard=tier(
                settings.standard_model, settings.standard_provider, settings.standard_kwargs,
                settings.standard_fallback_model, settings.standard_fallback_provider,
                settings.standard_timeout,
            ),
        )
