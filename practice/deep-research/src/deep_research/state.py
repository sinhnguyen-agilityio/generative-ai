"""Graph state, plus the structured outputs and tool schemas the agents' models produce."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ----- Graph state -----


class ResearchState(MessagesState):
    blocked: bool  # guardrail rejected the last message
    needs_clarification: bool  # scope asked the user a question
    brief: str
    criteria: list[str]  # acceptance criteria: what a complete report must answer (see verifier)
    plan: list[str]
    plan_feedback: str  # non-empty: the reviewer asked for a new plan
    supervisor_messages: Annotated[list[AnyMessage], add_messages]
    supervisor_iterations: int
    notes: Annotated[list[dict], operator.add]  # one topic note per researched topic (see researcher.finalize)
    report: str
    report_feedback: str  # non-empty: the user's follow-up message on the report
    research_request: str  # non-empty: the follow-up needs more research before the new report
    verification: dict | None  # see verifier.VerifierAgent.verify
    fix_rounds: int  # times the verifier sent the current report back to the writer
    evaluation: dict | None
    published_path: str


class SupervisorInput(TypedDict):
    brief: str
    plan: NotRequired[list[str]]
    research_request: str
    supervisor_messages: Annotated[list[AnyMessage], add_messages]
    supervisor_iterations: int


class SupervisorOutput(TypedDict):
    research_request: str
    supervisor_messages: Annotated[list[AnyMessage], add_messages]
    supervisor_iterations: int
    notes: Annotated[list[dict], operator.add]


class SupervisorState(SupervisorInput, SupervisorOutput):
    """Notes accumulate only within this invocation; input excludes the parent's old notes."""


class ResearcherState(TypedDict):
    topic: str
    researcher_messages: Annotated[list[AnyMessage], add_messages]
    sources: dict[str, dict]  # url -> {id, title, full_text, passages shown}: the numbers the report cites
    notes: dict  # the topic note: {"report": text, "sources": [{id, title, url, full_text, passages}]}


# ----- Structured outputs -----


class SafetyVerdict(BaseModel):
    allowed: bool = Field(description="True when the message may be researched.")
    reason: str = Field(description="Empty when allowed; otherwise one sentence on why it is blocked.")


class ClarifyWithUser(BaseModel):
    need_clarification: bool = Field(description="True if a question must be asked first.")
    question: str = Field(description="The question to ask; empty when none is needed.")
    verification: str = Field(description="Short note confirming research will start.")


class ResearchBrief(BaseModel):
    brief: str = Field(description="One detailed research question in the first person; no headings or sections.")
    criteria: list[str] = Field(
        description="2-5 acceptance criteria: each one short question the report must answer, taken from the research question."
    )


class FollowUp(BaseModel):
    kind: Literal["revise", "research"] = Field(
        description="revise: change how the report presents what it already has; research: needs facts it does not have."
    )


class ResearchPlan(BaseModel):
    topics: list[str] = Field(
        description="1-5 non-overlapping research topics covering the question; each one short, standalone sentence."
    )


class SentenceVerdict(BaseModel):
    sentence: int = Field(description="The sentence number n of S<n>.")
    verdict: Literal["supported", "unsupported", "no_claim"]
    problem: str = Field(description="For unsupported: what its cited evidence does not state, in one short sentence; else empty.")


class CriterionVerdict(BaseModel):
    criterion: int = Field(description="The criterion number n of C<n>.")
    met: bool = Field(description="True when the report answers it, or names it as not established.")
    problem: str = Field(description="When not met: what is missing, in one short sentence; else empty.")


class ReportCheck(BaseModel):
    sentences: list[SentenceVerdict] = Field(description="One verdict for every sentence, in order.")
    criteria: list[CriterionVerdict] = Field(description="One verdict for every acceptance criterion.")


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
