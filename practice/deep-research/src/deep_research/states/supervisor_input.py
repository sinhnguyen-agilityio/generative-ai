"""Supervisor input definitions."""

from __future__ import annotations

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class SupervisorInput(TypedDict):
    brief: str
    plan: NotRequired[list[str]]
    research_request: str
    supervisor_messages: Annotated[list[AnyMessage], add_messages]
    supervisor_iterations: int
