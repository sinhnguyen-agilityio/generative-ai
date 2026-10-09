"""Think definitions."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Think(BaseModel):
    """Pause to reflect on the latest results before the next step. Fetches nothing and changes
    nothing; the reflection only stays in the conversation to guide your next decision.

    When: before delegation if no initial plan exists, after each round of results arrives,
    and before deciding to stop. Use a turn of its own.

    Cover, briefly:
    1. Established: what the results answer, and from which sources.
    2. Missing: which parts of your task are still unanswered, and whether each is essential.
    3. Enough: whether the evidence answers the task, or the latest results added nothing new.
    4. Next: the exact next call (with its query or topic), or stop."""

    reflection: str = Field(description="Established, missing, enough, next: the four points above.")
