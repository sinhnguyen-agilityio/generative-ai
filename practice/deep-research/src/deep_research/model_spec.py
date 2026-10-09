"""Model spec definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from deep_research.utils.env import _env


@dataclass(frozen=True)
class ModelSpec:
    """One model tier with independent primary and fallback provider options."""

    model: str
    provider: str | None = None
    kwargs: dict = field(default_factory=dict)
    fallback_model: str | None = None
    fallback_provider: str | None = None
    timeout: float = 60.0
    fallback_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, tier: str, timeout: float) -> ModelSpec:
        """Load the primary and fallback configuration for a model tier.

        Args:
            tier: Tier suffix such as REASONING or STANDARD.
            timeout: Default request timeout in seconds.

        Returns:
            Model specification from INTELLIGENCE_<tier>_* variables.

        Raises:
            ValueError: The required model is missing or a numeric setting is invalid.
            json.JSONDecodeError: Model keyword arguments are not valid JSON.
        """
        prefix = f"INTELLIGENCE_{tier}_"
        model = _env(prefix + "MODEL")
        if not model:
            raise ValueError(f"{prefix}MODEL is required")
        return cls(
            model=model,
            provider=_env(prefix + "PROVIDER") or None,
            kwargs=json.loads(_env(prefix + "KWARGS") or "{}"),
            fallback_model=_env(prefix + "FALLBACK_MODEL") or None,
            fallback_provider=_env(prefix + "FALLBACK_PROVIDER") or None,
            timeout=float(_env(prefix + "TIMEOUT") or timeout),
            fallback_kwargs=json.loads(_env(prefix + "FALLBACK_KWARGS") or "{}"),
        )
