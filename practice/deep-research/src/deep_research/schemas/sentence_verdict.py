"""Sentence verdict definitions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SentenceVerdict(BaseModel):
    sentence: int = Field(description="The sentence number n of S<n>.")
    verdict: Literal["supported", "unsupported", "no_claim"]
    problem: str = Field(description="For unsupported: what its cited evidence does not state, in one short sentence; else empty.")
