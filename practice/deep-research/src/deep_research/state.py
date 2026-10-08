"""Graph state, plus the structured outputs and tool schemas the agents' models produce."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ----- Graph state -----


class ResearchState(MessagesState):
    blocked: bool  # guardrail rejected the last message
    needs_clarification: bool  # scope asked the user a question
    brief: str
    plan: list[str]
    plan_feedback: str  # non-empty: the reviewer asked for a new plan
    supervisor_messages: Annotated[list[AnyMessage], add_messages]
    supervisor_iterations: int
    notes: Annotated[list[dict], operator.add]  # one topic note per researched topic (see researcher.finalize)
    report: str
    report_feedback: str  # non-empty: the reviewer asked for a new report
    evaluation: dict | None
    published_path: str


class ResearcherState(TypedDict):
    topic: str
    researcher_messages: Annotated[list[AnyMessage], add_messages]
    searches: int
    sources: dict[str, dict]  # url -> {id, title, full_text}: the source numbers cited in the report
    notes: dict  # the topic note: {"report": text, "sources": [{id, title, url, full_text}]}


# ----- Structured outputs -----


class SafetyVerdict(BaseModel):
    allowed: bool = Field(description="True when the message may be researched.")
    reason: str = Field(description="Empty when allowed; otherwise one sentence on why it is blocked.")


class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(description="True if a question must be asked first.")
    question: str = Field(description="The question to ask; empty when none is needed.")
    verification: str = Field(description="Short note confirming research will start.")


class ResearchBrief(BaseModel):
    brief: str = Field(description="Self-contained research brief written from the conversation.")


class ResearchPlan(BaseModel):
    topics: list[str] = Field(
        description="2-5 independent research topics covering the brief; each one short, standalone sentence."
    )


class Evaluation(BaseModel):
    faithfulness: float = Field(ge=0, le=1, description="Share of report claims supported by the findings.")
    relevance: float = Field(ge=0, le=1, description="How well the report answers the brief.")
    comment: str = Field(description="One or two sentences explaining the scores.")


class Answered(BaseModel):
    question: str = Field(description="A sub-question of the topic that the results answer.")
    citations: list[int] = Field(description="Source numbers that support the answer.")


class MissingPart(BaseModel):
    question: str = Field(description="A sub-question of the topic that the results do not answer.")
    essential: bool = Field(
        description="True only if the topic explicitly asks for it and the results give no answer to it at all."
    )
    reason: str = Field(description="Not found, sources disagree, or only a snippet was available.")


class TopicReport(BaseModel):
    findings: str = Field(description="Markdown findings citing source numbers as [1]; no source list.")
    answered: list[Answered]
    missing_parts: list[MissingPart]
    coverage: Literal["full", "partial", "minimal"]


# ----- Tools (schemas only; the agents execute them) -----


class ConductResearch(BaseModel):
    """Delegate one research topic to a researcher. Give a complete, standalone description."""

    research_topic: str


class ResearchComplete(BaseModel):
    """Call when the collected findings are enough to write the report."""


class WebSearch(BaseModel):
    """Search the web. Returns the most relevant passages of the result pages, numbered by source."""

    query: str


class Think(BaseModel):
    """Pause to reflect on the latest results before the next step. Fetches nothing and changes
    nothing; the reflection only stays in the conversation to guide your next decision.

    When: after each round of results arrives, in a turn of its own, and before deciding to stop.

    Cover, briefly:
    1. Established: what the results answer, and from which sources.
    2. Missing: which parts of your task are still unanswered, and whether each is essential.
    3. Enough: whether the evidence answers the task, or the latest results added nothing new.
    4. Next: the exact next call (with its query or topic), or stop."""

    reflection: str = Field(description="Established, missing, enough, next: the four points above.")
