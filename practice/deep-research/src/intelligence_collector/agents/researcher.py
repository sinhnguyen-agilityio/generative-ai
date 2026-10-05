"""Research sub-agent: an LLM tool-calling loop over ``tavily_search`` and ``think_tool``.

Each sub-agent investigates exactly one research question in an isolated context::

    START -> llm_call --(tool calls, budget left)--> tool_node --> llm_call
                     \\--(final answer or budget spent)--> compress_research -> END

* ``llm_call``          LLM decision: search, reflect with ``think_tool``, or give a final note.
* ``tool_node``         Runs the searches through the MCP boundary and summarizes each fetched page
                        (``page_summarizer.py``); records reflections.
* ``compress_research`` Structured LLM call that condenses the page summaries into cited findings.
* ``route_after_llm``   Continues only while the LLM asks and budgets remain.

The LLM only ever reads page summaries; each page's full text travels in ``sources`` to
``worker.py``, which stores it in the VFS and indexes its chunks. Reflections stay in the message
history and never reach the compressor.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from intelligence_collector.agents.page_summarizer import PageSummarizer
from intelligence_collector.agents.prompts import (
    COMPRESS_TEMPLATE,
    RESEARCHER_TEMPLATE,
)
from intelligence_collector.agents.tools import (
    SEARCH_TOOL,
    SEARCH_TOOL_NAME,
    THINK_TOOL,
    THINK_TOOL_NAME,
    think,
)
from intelligence_collector.contracts import ResearchFindings
from intelligence_collector.observability.tracing import traced
from intelligence_collector.tools.mcp import MCPResearchPort, redact_mcp_error
from intelligence_collector.tools.url_tracking import UrlClaims, normalize_url

HARD_SEARCH_LIMIT = 5
# Distinct pages one sub-agent may keep for its single research question (todo), across all of
# its searches — not per search call.
DEFAULT_MAX_SOURCES = 5
HARD_SOURCE_LIMIT = 10
# Page summaries are bounded by the summarizer's output cap; this only guards against snippets
# or summaries that run long.
TOOL_RESULT_CHARS = 3000
COMPRESS_SOURCE_CHARS = 4000


class ResearcherState(TypedDict, total=False):
    """State of one research sub-agent's own LangGraph loop (LLM and tools, then compression)."""

    research_question: str
    messages: Annotated[list[BaseMessage], add_messages]
    llm_calls: int
    search_calls: int
    sources: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    findings: ResearchFindings | None


@dataclass
class ResearchOutcome:
    """Everything one sub-agent gathered for its research question."""

    question: str
    search_budget: int
    mcp_server: str
    mcp_tool: str
    search_calls: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)
    findings: ResearchFindings | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def unique_sources(self) -> list[dict[str, Any]]:
        """This outcome's sources with duplicate URLs removed."""
        return unique_sources(self.sources)


def unique_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first result per normalized URL; later copies of the same page are dropped."""
    seen, unique = set(), []
    for source in sources:
        identity = normalize_url(source.get("url", ""))
        if identity not in seen:
            seen.add(identity)
            unique.append(source)
    return unique


def _format_search_results(query: str, rows: list[dict[str, Any]], repeated: list[str]) -> str:
    if not rows:
        blocks = [f'No new results for "{query}".']
    else:
        blocks = [f'Search results for "{query}":']
        for number, row in enumerate(rows, start=1):
            linked = (
                " (already retrieved by another research task: linked, does not use your "
                "source budget)"
                if row.get("_covered_by")
                else ""
            )
            blocks.append(
                f"[{number}] {row.get('title') or 'Untitled'}{linked}\n"
                f"URL: {row.get('url', '')}\n"
                f"{str(row.get('content', ''))[:TOOL_RESULT_CHARS]}"
            )
    if repeated:
        blocks.append(
            "Already retrieved earlier, not repeated: "
            + ", ".join(repeated)
            + ". Change the query if you need new sources."
        )
    if any(row.get("_covered_by") for row in rows):
        blocks.append(
            "Pages marked as linked were found by another research task; prefer searching for "
            "sources specific to your own question."
        )
    return "\n\n".join(blocks)


def _format_sources_for_compression(sources: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        f"Source {number}: {source.get('title') or 'Untitled'}\n"
        f"URL: {source.get('url', '')}\n"
        f"Found by query: {source.get('_query', '')}\n"
        f"{str(source.get('content', ''))[:COMPRESS_SOURCE_CHARS]}"
        for number, source in enumerate(sources, start=1)
    )


class ResearchAgent:
    """Runs one isolated research sub-agent per research question."""

    def __init__(
        self,
        router,
        mcp_factory: Callable[[], MCPResearchPort],
        *,
        max_search_calls: int = HARD_SEARCH_LIMIT,
        max_sources: int = DEFAULT_MAX_SOURCES,
        page_summarizer: PageSummarizer | None = None,
    ):
        self.router = router
        self.mcp_factory = mcp_factory
        self.max_search_calls = min(max(1, max_search_calls), HARD_SEARCH_LIMIT)
        self.max_sources = min(max(1, max_sources), HARD_SOURCE_LIMIT)
        self.page_summarizer = page_summarizer or PageSummarizer(router)

    def research(
        self,
        question: str,
        *,
        run_id: str = "",
        thread_id: str = "",
        task_id: str = "",
        claims: UrlClaims | None = None,
    ) -> ResearchOutcome:
        """Run one sub-agent for `question`: search through MCP within the search and source
        budgets, then compress what was found into ``ResearchFindings``. `claims` is shared by
        parallel sub-agents so each page is captured by one task only.
        """
        # A fresh MCP client per sub-agent keeps search budgets and request logs per task.
        mcp = self.mcp_factory()
        budget = min(self.max_search_calls, getattr(mcp, "max_calls", HARD_SEARCH_LIMIT))
        outcome = ResearchOutcome(
            question=question,
            search_budget=budget,
            mcp_server=getattr(mcp, "server_name", "controlled"),
            mcp_tool=getattr(mcp, "tool_name", "search"),
        )
        graph = self._build_graph(
            mcp,
            outcome,
            run_id=run_id,
            thread_id=thread_id,
            task_id=task_id,
            claims=claims,
        )
        final = graph.invoke(
            {
                "research_question": question,
                "messages": [HumanMessage(content=f"Research question:\n{question}")],
                "llm_calls": 0,
                "search_calls": 0,
            },
            config={
                "run_name": "research_agent",
                "metadata": {"todo_id": task_id, "run_id": run_id},
            },
        )
        outcome.search_calls = final.get("search_calls", 0)
        outcome.sources = final.get("sources", [])
        outcome.errors = final.get("errors", [])
        outcome.findings = final.get("findings")
        return outcome

    def _build_graph(
        self,
        mcp: MCPResearchPort,
        outcome: ResearchOutcome,
        *,
        run_id,
        thread_id,
        task_id="",
        claims: UrlClaims | None = None,
    ):
        budget = outcome.search_budget
        source_budget = self.max_sources
        # One reflection per search, plus a couple of spare turns (e.g. a malformed tool call)
        # without looping forever.
        max_llm_calls = 2 * budget + 2
        # URLs this sub-agent has kept; repeats are not re-shown or re-collected.
        seen_urls: set[str] = set()
        # The subset this task retrieved first; only these count toward its source budget.
        owned_urls: set[str] = set()

        def llm_call(state: ResearcherState) -> ResearcherState:
            """Ask the researcher LLM for its next step: search, reflect, or give a final note."""
            messages = RESEARCHER_TEMPLATE.format_messages(
                messages=state["messages"],
                date=date.today().isoformat(),
                max_search_calls=budget,
                max_sources=source_budget,
            )
            response = self.router.invoke_with_tools(
                "researcher",
                messages,
                [SEARCH_TOOL, THINK_TOOL],
                run_id=run_id,
                thread_id=thread_id,
                # Without a forced first search, lighter models answer from prior knowledge (or
                # only reflect) and the task fails with no sources.
                tool_choice=SEARCH_TOOL_NAME if not state.get("search_calls") else None,
            )
            return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

        def keep_new_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
            """Split search rows into pages new to this task and URLs it already has. A page
            another task claimed first is linked (no raw page, no source slot); pages beyond the
            source budget are dropped."""
            fresh: list[dict[str, Any]] = []
            repeated: list[str] = []
            for row in rows:
                url = str(row.get("url", ""))
                key = normalize_url(url)
                if key in seen_urls:
                    repeated.append(url)
                    continue
                has_room = len(owned_urls) < source_budget
                if claims is None:
                    owner = task_id if has_room else None
                elif has_room:
                    owner = claims.claim(url, task_id)
                else:
                    # Full: never claim a page this task would then drop.
                    owner = claims.owner(url)
                if owner is not None and owner != task_id:
                    seen_urls.add(key)
                    fresh.append({**row, "raw_content": "", "_covered_by": owner})
                elif owner == task_id and has_room:
                    seen_urls.add(key)
                    owned_urls.add(key)
                    fresh.append(row)
            return fresh, repeated

        def read_owned_pages(question: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Summarize the pages this task owns; linked pages keep their snippet."""
            owned = [row for row in rows if not row.get("_covered_by")]
            read = iter(
                self.page_summarizer.read_pages(question, owned, run_id=run_id, thread_id=thread_id)
            )
            return [row if row.get("_covered_by") else next(read) for row in rows]

        def search(question: str, query: str) -> tuple[str, list[dict[str, Any]], list[str]]:
            """One search: returns the tool reply, the new sources and any error."""
            remaining = source_budget - len(owned_urls)
            try:
                with traced(
                    "mcp.search",
                    as_type="tool",
                    input={"query": query, "max_results": remaining},
                    metadata={
                        "mcp_server": outcome.mcp_server,
                        "mcp_tool": outcome.mcp_tool,
                        "todo_id": task_id,
                    },
                ) as observation:
                    rows = mcp.search(
                        query,
                        server=outcome.mcp_server,
                        tool=outcome.mcp_tool,
                        max_results=remaining,
                    )
                    # Locators only: page content stays in the VFS captures.
                    observation.output = {
                        "results": len(rows),
                        "urls": [str(row.get("url", "")) for row in rows],
                    }
            except Exception as exc:  # retrieval failures are task outcomes, not crashes
                error = redact_mcp_error(exc)
                return f"Search failed: {error}", [], [error]
            fresh, repeated = keep_new_rows(rows)
            fresh = read_owned_pages(question, fresh)
            content = _format_search_results(query, fresh, repeated)
            if len(owned_urls) >= source_budget:
                content += (
                    f"\n\nSource budget reached ({source_budget} pages). "
                    "Finish with the evidence you already have."
                )
            return content, [{**row, "_query": query} for row in fresh], []

        def tool_node(state: ResearcherState) -> ResearcherState:
            """Run each requested search or reflection within the budgets."""
            calls = state.get("search_calls", 0)
            replies: list[BaseMessage] = []
            sources: list[dict[str, Any]] = []
            errors: list[str] = []
            last = cast(AIMessage, state["messages"][-1])  # routed here only with tool calls
            for call in last.tool_calls:
                query = str(call["args"].get("query", "")).strip()
                if call["name"] == THINK_TOOL_NAME:
                    content = think(str(call["args"].get("reflection", "")))
                elif call["name"] != SEARCH_TOOL_NAME:
                    content = (
                        f"Unknown tool {call['name']!r}; use {SEARCH_TOOL_NAME} or "
                        f"{THINK_TOOL_NAME}."
                    )
                elif not query:
                    content = "A non-empty query is required."
                elif calls >= budget:
                    content = "Search budget exhausted. Finish with the evidence you already have."
                elif len(owned_urls) >= source_budget:
                    content = (
                        f"Source budget reached ({source_budget} pages). "
                        "Finish with the evidence you already have."
                    )
                else:
                    calls += 1
                    content, found, failed = search(state["research_question"], query)
                    sources.extend(found)
                    errors.extend(failed)
                replies.append(ToolMessage(content=content, tool_call_id=call["id"]))
            return {
                "messages": replies,
                "search_calls": calls,
                "sources": sources,
                "errors": errors,
            }

        def compress_research(state: ResearcherState) -> ResearcherState:
            """Compress the gathered page summaries into ``ResearchFindings``; reflections are
            not evidence and are left out. No sources means no findings.
            """
            if not state.get("sources"):
                return {"findings": None}
            last = state["messages"][-1]
            note = "" if getattr(last, "tool_calls", None) else last.text
            messages = COMPRESS_TEMPLATE.format_messages(
                question=state["research_question"],
                note=note or "(none)",
                results=_format_sources_for_compression(unique_sources(state["sources"])),
            )
            try:
                findings = self.router.invoke_messages(
                    "research_compressor",
                    messages,
                    ResearchFindings,
                    run_id=run_id,
                    thread_id=thread_id,
                )
            except Exception as exc:
                return {"findings": None, "errors": [f"Compression failed: {exc}"]}
            return {"findings": findings}

        def route_after_llm(state: ResearcherState) -> str:
            """Continue while the LLM asks for a tool and budgets remain; otherwise compress."""
            last = state["messages"][-1]
            wants_tool = isinstance(last, AIMessage) and bool(last.tool_calls)
            if (
                wants_tool
                and state.get("search_calls", 0) < budget
                and sum(not s.get("_covered_by") for s in state.get("sources", [])) < source_budget
                and state.get("llm_calls", 0) < max_llm_calls
            ):
                return "tool_node"
            return "compress_research"

        graph = StateGraph(ResearcherState)
        graph.add_node("llm_call", llm_call)
        graph.add_node("tool_node", tool_node)
        graph.add_node("compress_research", compress_research)
        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges("llm_call", route_after_llm, ["tool_node", "compress_research"])
        graph.add_edge("tool_node", "llm_call")
        graph.add_edge("compress_research", END)
        return graph.compile()
