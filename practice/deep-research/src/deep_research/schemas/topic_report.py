"""Topic report definitions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from deep_research.schemas.answered import Answered
from deep_research.schemas.missing_part import MissingPart


class TopicReport(BaseModel):
    findings: str = Field(description="Markdown findings citing source numbers as [1]; no source list.")
    answered: list[Answered]
    missing_parts: list[MissingPart]
    coverage: Literal["full", "partial", "minimal"]
