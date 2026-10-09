"""Follow up definitions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FollowUp(BaseModel):
    kind: Literal["revise", "research"] = Field(
        description="revise: change how the report presents what it already has; research: needs facts it does not have."
    )
