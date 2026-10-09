"""Safety verdict definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SafetyVerdict(BaseModel):
    allowed: bool = Field(description="True when the message may be researched.")
    reason: str = Field(description="Empty when allowed; otherwise one sentence on why it is blocked.")
