"""Researcher agent: researches one topic with web search, then writes a topic report for the
supervisor. It runs as its own small graph:

    researcher ⇄ researcher_tools → report

Each WebSearch indexes the result pages in the store and shows the model only the passages most
relevant to its query, numbered by source. The report's status and citations are checked in code.
"""

from __future__ import annotations

import logging
import re
from typing import Literal
from urllib.parse import urlsplit

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from deep_research.agents.utils import thread_id, today
from deep_research.config import Settings
from deep_research.prompts import RESEARCHER_PROMPT, TOPIC_REPORT_PROMPT
from deep_research.search import format_passages, prepare_pages
from deep_research.state import MissingPart, ResearcherState, Think, TopicReport, WebSearch

logger = logging.getLogger(__name__)

SMALL_RESULTS_CHARS = 8000  # results this short are shown whole, without retrieval
_CITATION = re.compile(r"\[(\d+)\]")


class ResearcherAgent:
    def __init__(self, llm, search, store, settings: Settings):
        self.llm = llm.tools([WebSearch, Think])
        self.reporter = llm.structured(TopicReport)
        self.search = search
        self.store = store
        self.max_searches = settings.max_researcher_searches
        self.retrieval_k = settings.retrieval_k
        self.graph = self._build()

    def research(self, topic: str) -> str:
        """The topic report; a failure is reported as a failed report, not raised."""
        try:
            result = self.graph.invoke(
                {"topic": topic, "researcher_messages": [HumanMessage(topic)], "searches": 0, "sources": {}},
                {"recursion_limit": 50},
            )
            return result["notes"]
        except Exception as exc:
            logger.exception("Researcher failed on %r", topic)
            return failed_report(f"Research failed: {exc}")

    # ----- graph nodes -----

    def researcher(self, state: ResearcherState):
        system = SystemMessage(RESEARCHER_PROMPT.format(date=today(), max_searches=self.max_searches))
        return {"researcher_messages": [self.llm.invoke([system, *state["researcher_messages"]])]}

    def researcher_tools(self, state: ResearcherState, config: RunnableConfig):
        searches, sources, replies = state["searches"], dict(state["sources"]), []
        for call in state["researcher_messages"][-1].tool_calls:
            if call["name"] == "Think":
                content = "Reflection recorded."
            elif call["name"] != "WebSearch":
                content = f"Unknown tool: {call['name']}"
            elif searches >= self.max_searches:
                content = "Search limit reached. Stop searching."
            else:
                searches += 1
                content = self._web_search(call["args"]["query"], state["topic"], thread_id(config), sources)
            replies.append(ToolMessage(content, tool_call_id=call["id"], name=call["name"]))
        return {"researcher_messages": replies, "searches": searches, "sources": sources}

    def report(self, state: ResearcherState):
        prompt = HumanMessage(TOPIC_REPORT_PROMPT.format(topic=state["topic"]))
        try:
            result = self.reporter.invoke([*state["researcher_messages"], prompt])
        except Exception as exc:
            logger.exception("Topic report failed on %r", state["topic"])
            return {"notes": failed_report(f"Topic report failed: {exc}")}
        return {"notes": finalize(result, state["sources"])}

    # ----- helpers -----

    def _web_search(self, query: str, topic: str, run_id: str, sources: dict[str, dict]) -> str:
        """Search, index the result pages, and return the passages most relevant to the query.
        New pages get the next source number in ``sources``; a page seen before keeps its number."""
        try:
            pages = prepare_pages(self.search.search(query))
        except Exception as exc:
            logger.exception("Search failed")
            return f"Search failed: {exc}"
        for page in pages:
            sources.setdefault(page["url"], {"id": len(sources) + 1, "title": page["title"], "full_text": page["full_text"]})
        return format_passages(self._passages(query, pages, topic, run_id), sources)

    def _passages(self, query: str, pages: list[dict], topic: str, run_id: str) -> list[dict]:
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
        return "researcher_tools" if state["researcher_messages"][-1].tool_calls else "report"

    def _after_tools(self, state: ResearcherState) -> Literal["researcher", "report"]:
        return "report" if state["searches"] >= self.max_searches else "researcher"

    def _build(self):
        builder = StateGraph(ResearcherState)
        builder.add_node("researcher", self.researcher)
        builder.add_node("researcher_tools", self.researcher_tools)
        builder.add_node("report", self.report)
        builder.add_edge(START, "researcher")
        builder.add_conditional_edges("researcher", self._after_researcher)
        builder.add_conditional_edges("researcher_tools", self._after_tools)
        builder.add_edge("report", END)
        # Several researchers run inside one supervisor step, so they cannot share its checkpoints.
        return builder.compile(checkpointer=False, name="researcher")


# ----- topic report -----


def finalize(report: TopicReport, sources: dict[str, dict]) -> str:
    """Check the model's report against the sources and render it for the supervisor.

    Citations to unknown source numbers are dropped; an answer left without a valid citation
    becomes an essential missing part; coverage cannot be full while an essential part is missing;
    the status is derived here, never taken from the model."""
    by_id = {s["id"]: (url, s) for url, s in sources.items()}
    findings = _CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in by_id else "", report.findings).strip()
    answered, missing = [], list(report.missing_parts)
    for item in report.answered:
        ids = [n for n in dict.fromkeys(item.citations) if n in by_id]
        if ids:
            answered.append((item.question, ids))
        else:
            missing.append(MissingPart(question=item.question, essential=True, reason="no valid source"))
    essential_missing = any(part.essential for part in missing)
    coverage = report.coverage
    if not answered:
        coverage = "minimal"
    elif coverage == "full" and essential_missing:
        coverage = "partial"
    cited = sorted({int(n) for n in _CITATION.findall(findings)} | {n for _, ids in answered for n in ids})
    if not cited:
        status = "failed"
    elif coverage == "full" and not essential_missing:
        status = "complete"
    else:
        status = "partial"

    lines = [f"## Status: {status}", f"## Coverage: {coverage}", "Answered:"]
    lines += [f"- {q} {''.join(f'[{n}]' for n in ids)}" for q, ids in answered] or ["- None"]
    lines.append("## Missing parts")
    lines += [
        f"- {p.question} ({'essential' if p.essential else 'not essential'}: {p.reason})" for p in missing
    ] or ["- None"]
    lines += ["## Findings", findings or "None.", "## Citations"]
    domains = {urlsplit(by_id[n][0]).netloc for n in cited}
    snippets = sum(not by_id[n][1]["full_text"] for n in cited)
    lines.append(f"{len(cited)} sources, {len(domains)} domains, {snippets} snippet only")
    lines += [f"[{n}] {by_id[n][1]['title']}: {by_id[n][0]}" for n in cited]
    return "\n".join(lines)


def failed_report(reason: str) -> str:
    return "\n".join(
        ["## Status: failed", "## Coverage: minimal", "## Missing parts", f"- The whole topic (essential: {reason})"]
    )
