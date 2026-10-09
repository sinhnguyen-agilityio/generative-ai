"""Criterion verdict definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CriterionVerdict(BaseModel):
    criterion: int = Field(description="The criterion number n of C<n>.")
    met: bool = Field(description="True when the report answers it, or names it as not established.")
    problem: str = Field(description="When not met: what is missing, in one short sentence; else empty.")
