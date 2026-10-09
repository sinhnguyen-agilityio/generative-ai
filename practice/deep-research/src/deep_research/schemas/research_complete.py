"""Research complete definitions."""

from __future__ import annotations

from pydantic import BaseModel


class ResearchComplete(BaseModel):
    """Call when the collected findings are enough to write the report."""
