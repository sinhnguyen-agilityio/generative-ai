"""Report check definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field

from deep_research.schemas.criterion_verdict import CriterionVerdict
from deep_research.schemas.sentence_verdict import SentenceVerdict


class ReportCheck(BaseModel):
    sentences: list[SentenceVerdict] = Field(description="One verdict for every sentence, in order.")
    criteria: list[CriterionVerdict] = Field(description="One verdict for every acceptance criterion.")
