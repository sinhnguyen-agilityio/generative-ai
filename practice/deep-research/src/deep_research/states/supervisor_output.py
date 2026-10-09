"""Supervisor output definitions."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class SupervisorOutput(TypedDict):
    research_request: str
    supervisor_messages: Annotated[list[AnyMessage], add_messages]
    supervisor_iterations: int
    notes: Annotated[list[dict], operator.add]
