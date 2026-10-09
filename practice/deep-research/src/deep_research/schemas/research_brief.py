"""Research brief definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ResearchBrief(BaseModel):
    brief: str = Field(description="One detailed research question in the first person; no headings or sections.")
    criteria: list[str] = Field(
        description="2-5 acceptance criteria: each one short question the report must answer, taken from the research question."
    )
