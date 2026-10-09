"""Read stripped environment values and boolean flags."""

from __future__ import annotations

import os


def _env(name: str, default: str = "") -> str:
    """Read an environment value and strip surrounding whitespace.

    Args:
        name: Environment variable name.
        default: Value used when the variable is absent.

    Returns:
        Stripped environment value or default.
    """
    return os.getenv(name, default).strip()


def _flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment flag using the supported truthy values.

    Args:
        name: Environment variable name.
        default: Boolean used when the variable is absent.

    Returns:
        True for 1, true, yes, or on, ignoring case; otherwise False.
    """
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}
