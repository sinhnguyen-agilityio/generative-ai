"""Thread view definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ThreadView:
    values: dict
    pending: dict | None  # the interrupt payload waiting for a human decision
    unfinished: bool  # stopped mid-run (error or closed tab); can be continued
