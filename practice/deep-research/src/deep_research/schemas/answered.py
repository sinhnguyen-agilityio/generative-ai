"""Answered definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Answered(BaseModel):
    question: str = Field(description="A sub-question of the topic that the results answer.")
    citations: list[int] = Field(description="Source numbers that support the answer.")
