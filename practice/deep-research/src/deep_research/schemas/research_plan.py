"""Research plan definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchPlan(BaseModel):
    topics: list[str] = Field(
        description="1-5 non-overlapping research topics covering the question; each one short, standalone sentence."
    )
