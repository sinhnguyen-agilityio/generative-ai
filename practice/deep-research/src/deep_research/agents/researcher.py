"""Researcher agent: researches one topic with web search, then writes a topic report for the
supervisor. It runs as its own small graph:

    researcher ⇄ researcher_tools → report

Each WebSearch indexes the result pages in the store and shows the model only the passages most
relevant to its query, numbered by source. The report's status and citations are checked in code.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, StructuredTool
from langgraph.graph import END, START, StateGraph

from deep_research.prompts import RESEARCHER_PROMPT, TOPIC_REPORT_PROMPT
from deep_research.schemas.think import Think
from deep_research.schemas.topic_report import TopicReport
from deep_research.settings import Settings
from deep_research.states.researcher_state import ResearcherState
from deep_research.utils.notes import failed_note, finalize
from deep_research.utils.runtime import thread_id
from deep_research.utils.search import format_passages, prepare_pages
from deep_research.utils.text import today

logger = logging.getLogger(__name__)


SMALL_RESULTS_CHARS = 8000  # results this short are shown whole, without retrieval


class ResearcherAgent:
    def __init__(self, llm, search, store, settings: Settings):
        """Configure the executable search tool, topic reporter, and isolated research graph.

        Args:
            llm: Model tier for search decisions and structured topic reporting.
            search: Asynchronous web search backend.
            store: Synchronous page indexing and passage retrieval backend.
            settings: Search-budget prompt guidance and passage retrieval count.
        """
        search_tool = StructuredTool.from_function(
            coroutine=self._web_search, name="WebSearch",
            description="Search the web and return relevant source passages, numbered for citations.",
        )
        self.search_tool = search_tool
        self.llm = llm.tools([self.search_tool, Think])
        self.reporter = llm.structured(TopicReport)
        self.search = search
        self.store = store
        self.max_searches = settings.max_researcher_searches
        self.retrieval_k = settings.retrieval_k
        self.graph = self._build()

    async def research(self, topic: str) -> dict:
        """Run an isolated research loop for one delegated topic.

        Args:
            topic: Standalone research task supplied by the supervisor.
        Returns:
            A topic note with findings and sources, or a failed note on error.
        """
        try:
            result = await self.graph.ainvoke(
                {"topic": topic, "researcher_messages": [HumanMessage(topic)], "sources": {}},
            )
            return result["notes"]
        except Exception as exc:
            logger.exception("Researcher failed on %r", topic)
            return failed_note(f"Research failed: {exc}")

    # ----- graph nodes -----

    async def researcher(self, state: ResearcherState):
        """Ask the model to search, reflect, or finish gathering evidence.

        Args:
            state: Topic and the search/reflection conversation so far.
        Returns:
            The model response appended to researcher messages.
        """
        system = SystemMessage(RESEARCHER_PROMPT.format(date=today(), max_searches=self.max_searches))
        return {"researcher_messages": [await self.llm.ainvoke([system, *state["researcher_messages"]])]}

    async def researcher_tools(self, state: ResearcherState, config: RunnableConfig):
        """Run searches from one model turn concurrently and acknowledge Think.

        Args:
            state: Latest tool calls and the topic's source registry.
            config: Graph configuration identifying the research thread.
        Returns:
            Tool replies and sources with the passages shown to the researcher.
        """
        sources = dict(state["sources"])
        calls = state["researcher_messages"][-1].tool_calls
        inputs = [
            {**call, "args": {**call["args"], "state": {**state, "sources": sources}}}
            for call in calls if call["name"] == "WebSearch"
        ]
        results = await asyncio.gather(*(self.search_tool.ainvoke(call, config) for call in inputs))
        by_id = {reply.tool_call_id: reply for reply in results}
        replies = []
        for call in calls:
            if call["name"] == "WebSearch":
                replies.append(by_id[call["id"]])
            else:
                content = "Reflection recorded." if call["name"] == "Think" else f"Unknown tool: {call['name']}"
                replies.append(ToolMessage(content, tool_call_id=call["id"], name=call["name"]))
        return {"researcher_messages": replies, "sources": sources}

    async def report(self, state: ResearcherState):
        """Turn collected evidence into a structured, citation-checked topic report.

        Args:
            state: Research conversation, topic, and source registry.
        Returns:
            A finalized topic note, or a failed note if report generation fails.
        """
        prompt = HumanMessage(TOPIC_REPORT_PROMPT.format(topic=state["topic"]))
        try:
            result = await self.reporter.ainvoke([*state["researcher_messages"], prompt])
        except Exception as exc:
            logger.exception("Topic report failed on %r", state["topic"])
            return {"notes": failed_note(f"Topic report failed: {exc}")}
        return {"notes": finalize(result, state["sources"])}

    # ----- search tool and retrieval -----

    async def _web_search(self, query: str, state: Annotated[dict, InjectedToolArg], config: RunnableConfig) -> str:
        """Search, index the result pages, and return the passages most relevant to the query.
        New pages get the next source number in ``sources``; a page seen before keeps its number.
        Each source keeps the passages shown to the model: the evidence its citations rest on.

        Args:
            query: Model-generated web search query.
            state: Injected topic and source registry; hidden from the model.
            config: Injected graph configuration identifying the research thread.
        Returns:
            Formatted source passages, or a search failure message.
        """
        sources = state["sources"]
        try:
            pages = prepare_pages(await self.search.search(query))
        except Exception as exc:
            logger.exception("Search failed")
            return f"Search failed: {exc}"
        for page in pages:
            sources.setdefault(
                page["url"], {"id": len(sources) + 1, "title": page["title"], "full_text": page["full_text"], "passages": []}
            )
        passages = await asyncio.to_thread(self._passages, query, pages, state["topic"], thread_id(config))
        for passage in passages:
            source = sources[passage["url"]]
            sources[passage["url"]] = {**source, "passages": [*source["passages"], passage["text"]]}
        return format_passages(passages, sources)

    def _passages(self, query: str, pages: list[dict], topic: str, run_id: str) -> list[dict]:
        """Index pages and select evidence, falling back to snippets on retrieval errors.

        Args:
            query: Search query used to rank passages.
            pages: Prepared search results with page text and snippets.
            topic: Delegated task recorded in store metadata.
            run_id: Thread identifier used to isolate indexed evidence.
        Returns:
            URL/text passages to show to the researcher.
        """
        small = sum(len(p["text"]) for p in pages) < SMALL_RESULTS_CHARS
        try:
            self.store.add(pages, run_id=run_id, topic=topic)
            if not small:
                return self.store.query(query, run_id=run_id, urls=[p["url"] for p in pages], k=self.retrieval_k)
        except Exception:
            logger.exception("Indexing or retrieval failed")
            if not small:
                return [{"url": p["url"], "text": p["snippet"]} for p in pages if p["snippet"]]
        return [{"url": p["url"], "text": p["text"]} for p in pages]

    def _after_researcher(self, state: ResearcherState) -> Literal["researcher_tools", "report"]:
        """Route model tool calls to execution or a final response to report writing.

        Args:
            state: Research conversation containing the latest model response.

        Returns:
            researcher_tools when tool calls exist; otherwise report.
        """
        return "researcher_tools" if state["researcher_messages"][-1].tool_calls else "report"

    def _build(self):
        """Compile the researcher/tool/report loop without shared checkpoints.

        Returns:
            Researcher subgraph isolated from the parent supervisor checkpoint history.
        """
        builder = StateGraph(ResearcherState)
        builder.add_node("researcher", self.researcher)
        builder.add_node("researcher_tools", self.researcher_tools)
        builder.add_node("report", self.report)
        builder.add_edge(START, "researcher")
        builder.add_conditional_edges("researcher", self._after_researcher)
        builder.add_edge("researcher_tools", "researcher")
        builder.add_edge("report", END)
        # Several researchers run inside one supervisor step, so they cannot share its checkpoints.
        return builder.compile(checkpointer=False, name="researcher")
