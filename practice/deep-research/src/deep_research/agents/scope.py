"""Scope agent: asks one clarifying question when needed, then writes the research brief."""

from __future__ import annotations

from langchain_core.messages import AIMessage, get_buffer_string

from deep_research.agents.utils import today
from deep_research.prompts import BRIEF_PROMPT, CLARIFY_PROMPT
from deep_research.state import ClarifyWithUser, ResearchBrief, ResearchState


class ScopeAgent:
    def __init__(self, llm):
        self.clarifier = llm.structured(ClarifyWithUser)
        self.brief_writer = llm.structured(ResearchBrief)

    def clarify(self, state: ResearchState):
        prompt = CLARIFY_PROMPT.format(date=today(), messages=get_buffer_string(state["messages"]))
        result = self.clarifier.invoke(prompt)
        reply = result.question if result.need_clarification else result.verification
        return {"needs_clarification": result.need_clarification, "messages": [AIMessage(reply)]}

    def write_brief(self, state: ResearchState):
        prompt = BRIEF_PROMPT.format(date=today(), messages=get_buffer_string(state["messages"]))
        return {"brief": self.brief_writer.invoke(prompt).brief}
