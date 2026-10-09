"""Researcher agent: researches one topic with web search, then writes a topic report for the
supervisor. It runs as its own small graph:

    researcher ⇄ researcher_tools → report

Each WebSearch indexes the result pages in the store and shows the model only the passages most
relevant to its query, numbered by source. The report's status and citations are checked in code.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Literal
from urllib.parse import urlsplit

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from deep_research.agents.utils import CITATION, SENTENCE_END, thread_id, today
from deep_research.config import Settings
from deep_research.prompts import RESEARCHER_PROMPT, TOPIC_REPORT_PROMPT
from deep_research.search import format_passages, prepare_pages
from deep_research.state import MissingPart, ResearcherState, Think, TopicReport, WebSearch

logger = logging.getLogger(__name__)

SMALL_RESULTS_CHARS = 8000  # results this short are shown whole, without retrieval
_SOURCE_LINE = re.compile(r"^\[\d+\] .+: \S+$")


class ResearcherAgent:
    def __init__(self, llm, search, store, settings: Settings):
        self.llm = llm.tools([WebSearch, Think])
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
                {"recursion_limit": 50},
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
        searches = [call for call in calls if call["name"] == "WebSearch"]
        results = await asyncio.gather(*(
            self._web_search(call["args"]["query"], state["topic"], thread_id(config), sources)
            for call in searches
        ))
        contents = {call["id"]: result for call, result in zip(searches, results)}
        replies = []
        for call in calls:
            if call["name"] == "WebSearch":
                content = contents[call["id"]]
            elif call["name"] == "Think":
                content = "Reflection recorded."
            else:
                content = f"Unknown tool: {call['name']}"
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

    # ----- helpers -----

    async def _web_search(self, query: str, topic: str, run_id: str, sources: dict[str, dict]) -> str:
        """Search, index the result pages, and return the passages most relevant to the query.
        New pages get the next source number in ``sources``; a page seen before keeps its number.
        Each source keeps the passages shown to the model: the evidence its citations rest on.

        Args:
            query: Model-generated web search query.
            topic: Research task used as indexing metadata.
            run_id: Thread identifier for evidence retrieval.
            sources: Shared source registry updated with IDs and displayed passages.
        Returns:
            Formatted source passages, or a search failure message.
        """
        try:
            pages = prepare_pages(await self.search.search(query))
        except Exception as exc:
            logger.exception("Search failed")
            return f"Search failed: {exc}"
        for page in pages:
            sources.setdefault(
                page["url"], {"id": len(sources) + 1, "title": page["title"], "full_text": page["full_text"], "passages": []}
            )
        passages = await asyncio.to_thread(self._passages, query, pages, topic, run_id)
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
        """Route tool calls to execution; a response without tools starts report writing."""
        return "researcher_tools" if state["researcher_messages"][-1].tool_calls else "report"

    def _build(self):
        """Return the researcher/tools/report graph without shared checkpoints."""
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


# ----- topic report -----


def finalize(report: TopicReport, sources: dict[str, dict]) -> dict:
    """Check the model's report against the sources; return the topic note
    ``{"report": text, "sources": [{id, title, url, full_text, passages}]}`` with only the cited sources.

    The sources travel as data, never as text to parse back: the report quotes web content, so
    any source list written into it could be forged.

    A sentence that cites only unknown source numbers is dropped, and unknown numbers are removed
    from the others; an answer left without a valid citation becomes an essential missing part;
    a missing part listed twice is kept once (essential if either copy is); coverage cannot be full
    while an essential part is missing; the status is derived here, never taken from the model."""
    by_id = {s["id"]: (url, s) for url, s in sources.items()}
    findings = _keep_cited(report.findings, by_id.keys())
    answered, parts = [], list(report.missing_parts)
    for item in report.answered:
        ids = [n for n in dict.fromkeys(item.citations) if n in by_id]
        if ids:
            answered.append((item.question, ids))
        else:
            parts.append(MissingPart(question=item.question, essential=True, reason="no valid source"))
    unique: dict[str, MissingPart] = {}
    for part in parts:
        key = " ".join(part.question.lower().split())
        if key not in unique or (part.essential and not unique[key].essential):
            unique[key] = part
    missing = list(unique.values())
    essential_missing = any(part.essential for part in missing)
    coverage = report.coverage
    if not answered:
        coverage = "minimal"
    elif coverage == "full" and essential_missing:
        coverage = "partial"
    cited = sorted({int(n) for n in CITATION.findall(findings)} | {n for _, ids in answered for n in ids})
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
    lines += ["## Findings", findings or "None."]
    cited_sources = [
        {"id": n, "url": by_id[n][0], **{k: by_id[n][1][k] for k in ("title", "full_text", "passages")}}
        for n in cited
    ]
    return {"report": "\n".join(lines), "sources": cited_sources}


def render_note(note: dict) -> str:
    """The topic note as the supervisor reads it: the report plus its Citations section."""
    sources = note["sources"]
    if not sources:
        return note["report"]
    domains = {urlsplit(s["url"]).netloc for s in sources}
    snippets = sum(not s["full_text"] for s in sources)
    lines = [note["report"], "## Citations", f"{len(sources)} sources, {len(domains)} domains, {snippets} snippet only"]
    lines += [f"[{s['id']}] {s['title']}: {s['url']}" for s in sources]
    return "\n".join(lines)


def _keep_cited(findings: str, valid) -> str:
    lines = []
    for line in findings.splitlines():
        # Headings and source lines are written by code only; in findings they echo web content.
        if line.lstrip().startswith("## ") or _SOURCE_LINE.match(line.strip()):
            continue
        kept = []
        for sentence in SENTENCE_END.split(line):
            ids = [int(n) for n in CITATION.findall(sentence)]
            if ids and not any(n in valid for n in ids):
                continue
            kept.append(CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", sentence))
        if kept or not line.strip():
            lines.append(" ".join(kept))
    return "\n".join(lines).strip()


def failed_note(reason: str) -> dict:
    report = "\n".join(
        ["## Status: failed", "## Coverage: minimal", "## Missing parts", f"- The whole topic (essential: {reason})"]
    )
    return {"report": report, "sources": []}
