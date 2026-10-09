"""Clarify with user definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(description="True if a question must be asked first.")
    question: str = Field(description="The question to ask; empty when none is needed.")
    verification: str = Field(description="Short note confirming research will start.")
