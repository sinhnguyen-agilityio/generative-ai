"""Supervisor state definitions."""

from __future__ import annotations

from deep_research.states.supervisor_input import SupervisorInput
from deep_research.states.supervisor_output import SupervisorOutput


class SupervisorState(SupervisorInput, SupervisorOutput):
    """Notes accumulate only within this invocation; input excludes the parent's old notes."""
