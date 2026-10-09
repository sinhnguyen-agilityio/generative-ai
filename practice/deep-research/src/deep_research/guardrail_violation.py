"""Guardrail violation definitions."""

from __future__ import annotations


class GuardrailViolation(ValueError):
    """The input was rejected before reaching the graph; the message is safe to show."""
