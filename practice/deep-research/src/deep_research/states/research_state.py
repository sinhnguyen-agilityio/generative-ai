"""Research state definitions."""

from __future__ import annotations

import operator
from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages


class ResearchState(MessagesState):
    blocked: bool  # guardrail rejected the last message
    needs_clarification: bool  # scope asked the user a question
    brief: str
    criteria: list[str]  # acceptance criteria: what a complete report must answer (see verifier)
    plan: list[str]
    plan_feedback: str  # non-empty: the reviewer asked for a new plan
    supervisor_messages: Annotated[list[AnyMessage], add_messages]
    supervisor_iterations: int
    notes: Annotated[list[dict], operator.add]  # one topic note per researched topic (see utils.notes.finalize)
    report: str
    report_feedback: str  # non-empty: the user's follow-up message on the report
    research_request: str  # non-empty: the follow-up needs more research before the new report
    verification: dict | None  # see verifier.VerifierAgent.verify
    fix_rounds: int  # times the verifier sent the current report back to the writer
    evaluation: dict | None
    published_path: str
