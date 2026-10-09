"""Tier definitions."""

from __future__ import annotations

from deep_research.model_spec import ModelSpec
from deep_research.utils.models import _create, _structured


class Tier:
    """A primary chat model plus an optional fallback, tried in that order."""

    def __init__(self, spec: ModelSpec):
        """Initialize the primary model and optional fallback for a tier.

        Args:
            spec: Model identifiers, providers, options, and timeout.
        """
        self.primary = _create(spec.model, spec.provider, spec.timeout, spec.kwargs)
        self.fallback = (
            _create(spec.fallback_model, spec.fallback_provider, spec.timeout, {})
            if spec.fallback_model
            else None
        )

    def _wrap(self, build):
        """Apply the same runnable configuration to primary and fallback models.

        Args:
            build: Callable configuring a runnable from a chat model.

        Returns:
            Configured primary runnable with its fallback when available.
        """
        runnable = build(self.primary)
        return runnable.with_fallbacks([build(self.fallback)]) if self.fallback else runnable

    def structured(self, schema):
        """Configure structured output for both models in the tier.

        Args:
            schema: Schema for the model response.

        Returns:
            Structured-output runnable with optional fallback.
        """
        return self._wrap(lambda llm: _structured(llm, schema))

    def tools(self, tools):
        """Expose tool schemas to both models in the tier.

        Args:
            tools: Tool objects or schemas to bind; binding does not execute them.

        Returns:
            Tool-calling runnable with optional fallback.
        """
        return self._wrap(lambda llm: llm.bind_tools(tools))

    def plain(self):
        """Return the tier without tool or structured-output configuration.

        Returns:
            Chat model runnable with optional fallback.
        """
        return self._wrap(lambda llm: llm)
