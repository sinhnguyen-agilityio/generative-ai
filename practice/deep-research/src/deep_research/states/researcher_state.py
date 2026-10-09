"""Researcher state definitions."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class ResearcherState(TypedDict):
    topic: str
    researcher_messages: Annotated[list[AnyMessage], add_messages]
    sources: dict[str, dict]  # url -> {id, title, full_text, passages shown}: the numbers the report cites
    notes: dict  # the topic note: {"report": text, "sources": [{id, title, url, full_text, passages}]}
