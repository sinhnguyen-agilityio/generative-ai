"""Conduct research definitions."""

from __future__ import annotations

from pydantic import BaseModel


class ConductResearch(BaseModel):
    """Delegate one research topic to a researcher. Give a complete, standalone description."""

    research_topic: str
