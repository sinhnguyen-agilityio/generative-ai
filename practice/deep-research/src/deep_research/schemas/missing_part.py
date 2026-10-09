"""Missing part definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MissingPart(BaseModel):
    question: str = Field(description="A sub-question of the topic that the results do not answer.")
    essential: bool = Field(
        description="True only if the topic explicitly asks for it and the results give no answer to it at all."
    )
    reason: str = Field(description="Not found, sources disagree, or only a snippet was available.")
