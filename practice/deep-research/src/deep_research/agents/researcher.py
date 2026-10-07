"""Researcher agent: researches one topic with web search, then compresses what it found.
It runs as its own small graph:

    researcher ⇄ researcher_tools → compress
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from deep_research.agents.utils import thread_id, today
from deep_research.config import Settings
from deep_research.prompts import COMPRESS_PROMPT, RESEARCHER_PROMPT
from deep_research.search import format_results
from deep_research.state import ResearcherState, Think, WebSearch

logger = logging.getLogger(__name__)


class ResearcherAgent:
    def __init__(self, llm, search, store, settings: Settings):
        self.llm = llm.tools([WebSearch, Think])
        self.writer = llm.plain()
        self.search = search
        self.store = store
        self.max_searches = settings.max_researcher_searches
        self.graph = self._build()

    def research(self, topic: str) -> str:
        """The compressed findings for one topic; a failure is reported as text, not raised."""
        try:
            result = self.graph.invoke(
                {"topic": topic, "researcher_messages": [HumanMessage(topic)], "searches": 0},
                {"recursion_limit": 50},
            )
            return result["notes"]
        except Exception as exc:
            logger.exception("Researcher failed on %r", topic)
            return f"Research failed: {exc}"

    # ----- graph nodes -----

    def researcher(self, state: ResearcherState):
        system = SystemMessage(RESEARCHER_PROMPT.format(date=today(), max_searches=self.max_searches))
        return {"researcher_messages": [self.llm.invoke([system, *state["researcher_messages"]])]}

    def researcher_tools(self, state: ResearcherState, config: RunnableConfig):
        searches, replies = state["searches"], []
        for call in state["researcher_messages"][-1].tool_calls:
            if call["name"] == "Think":
                content = "Reflection recorded."
            elif call["name"] != "WebSearch":
                content = f"Unknown tool: {call['name']}"
            elif searches >= self.max_searches:
                content = "Search limit reached. Stop searching."
            else:
                searches += 1
                content = self._web_search(call["args"]["query"], state["topic"], thread_id(config))
            replies.append(ToolMessage(content, tool_call_id=call["id"], name=call["name"]))
        return {"researcher_messages": replies, "searches": searches}

    def compress(self, state: ResearcherState):
        messages = [
            SystemMessage(COMPRESS_PROMPT.format(topic=state["topic"])),
            *state["researcher_messages"],
            HumanMessage("Write the findings now."),
        ]
        return {"notes": self.writer.invoke(messages).text}

    # ----- helpers -----

    def _web_search(self, query: str, topic: str, run_id: str) -> str:
        try:
            results = self.search.search(query)
        except Exception as exc:
            logger.exception("Search failed")
            return f"Search failed: {exc}"
        self.store.add(results, run_id=run_id, topic=topic)
        return format_results(results)

    def _after_researcher(self, state: ResearcherState) -> Literal["researcher_tools", "compress"]:
        return "researcher_tools" if state["researcher_messages"][-1].tool_calls else "compress"

    def _after_tools(self, state: ResearcherState) -> Literal["researcher", "compress"]:
        return "compress" if state["searches"] >= self.max_searches else "researcher"

    def _build(self):
        builder = StateGraph(ResearcherState)
        builder.add_node("researcher", self.researcher)
        builder.add_node("researcher_tools", self.researcher_tools)
        builder.add_node("compress", self.compress)
        builder.add_edge(START, "researcher")
        builder.add_conditional_edges("researcher", self._after_researcher)
        builder.add_conditional_edges("researcher_tools", self._after_tools)
        builder.add_edge("compress", END)
        # Several researchers run inside one supervisor step, so they cannot share its checkpoints.
        return builder.compile(checkpointer=False, name="researcher")
