"""Research sub-agent: an LLM tool-calling loop over the Tavily MCP search tool.

Each sub-agent investigates exactly one research question in an isolated context::

    START -> llm_call --(tool calls, budget left)--> tool_node --> llm_call
                     \\--(final answer or budget spent)--> compress_research -> END

* ``llm_call``          LLM decision: call ``tavily_search`` or give a final note.
* ``tool_node``         Executes the requested searches through the MCP boundary.
* ``compress_research`` Structured LLM call that condenses the raw results into cited findings.
* ``route_after_llm``   Continues searching only while the LLM asks and budget remains.

The agent only gathers and compresses context; persistence belongs to ``worker.py``.
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

from intelligence_collector.agents.prompts import (
    COMPRESS_PROMPT_VERSION,
    COMPRESS_TEMPLATE,
    RESEARCHER_PROMPT_VERSION,
    RESEARCHER_TEMPLATE,
)
from intelligence_collector.contracts import ResearchFindings
from intelligence_collector.tools.mcp import MCPResearchPort, redact_mcp_error
from intelligence_collector.tools.url_tracking import normalize_url

HARD_SEARCH_LIMIT = 5
TOOL_RESULT_CHARS = 1500
COMPRESS_SOURCE_CHARS = 4000

SEARCH_TOOL_NAME = "tavily_search"
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": SEARCH_TOOL_NAME,
        "description": (
            "Search the web through the approved Tavily MCP server. Returns titles, URLs, and "
            "content excerpts for the most relevant pages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused web search query."},
            },
            "required": ["query"],
        },
    },
}


class ResearcherState(TypedDict, total=False):
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
            blocks.append(
                f"[{number}] {row.get('title') or 'Untitled'}\n"
                f"URL: {row.get('url', '')}\n"
                f"{str(row.get('content', ''))[:TOOL_RESULT_CHARS]}"
            )
    if repeated:
        blocks.append(
            "Already retrieved earlier, not repeated: " + ", ".join(repeated)
            + ". Change the query if you need new sources."
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
    ):
        self.router = router
        self.mcp_factory = mcp_factory
        self.max_search_calls = min(max(1, max_search_calls), HARD_SEARCH_LIMIT)

    def research(self, question: str, *, run_id: str = "", thread_id: str = "") -> ResearchOutcome:
        # A fresh MCP client per sub-agent keeps search budgets and request logs per task.
        mcp = self.mcp_factory()
        budget = min(self.max_search_calls, getattr(mcp, "max_calls", HARD_SEARCH_LIMIT))
        outcome = ResearchOutcome(
            question=question,
            search_budget=budget,
            mcp_server=getattr(mcp, "server_name", "controlled"),
            mcp_tool=getattr(mcp, "tool_name", "search"),
        )
        graph = self._build_graph(mcp, outcome, run_id=run_id, thread_id=thread_id)
        final = graph.invoke({
            "research_question": question,
            "messages": [HumanMessage(content=f"Research question:\n{question}")],
            "llm_calls": 0,
            "search_calls": 0,
        })
        outcome.search_calls = final.get("search_calls", 0)
        outcome.sources = final.get("sources", [])
        outcome.errors = final.get("errors", [])
        outcome.findings = final.get("findings")
        return outcome

    def _build_graph(self, mcp: MCPResearchPort, outcome: ResearchOutcome, *, run_id, thread_id):
        budget = outcome.search_budget
        # Allow a couple of non-search turns (e.g. a malformed tool call) without looping forever.
        max_llm_calls = budget + 2
        # URLs this sub-agent has already retrieved; repeats are not re-shown or re-collected.
        seen_urls: set[str] = set()

        def llm_call(state: ResearcherState) -> ResearcherState:
            messages = RESEARCHER_TEMPLATE.format_messages(
                messages=state["messages"],
                date=date.today().isoformat(),
                max_search_calls=budget,
            )
            response = self.router.invoke_with_tools(
                "researcher",
                messages,
                [SEARCH_TOOL],
                prompt_version=RESEARCHER_PROMPT_VERSION,
                run_id=run_id,
                thread_id=thread_id,
                # Without a forced first search, lighter models answer from prior knowledge and
                # the task fails with no sources.
                tool_choice="any" if not state.get("search_calls") else None,
            )
            return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

        def tool_node(state: ResearcherState) -> ResearcherState:
            calls = state.get("search_calls", 0)
            replies: list[BaseMessage] = []
            sources: list[dict[str, Any]] = []
            errors: list[str] = []
            last = cast(AIMessage, state["messages"][-1])  # routed here only with tool calls
            for call in last.tool_calls:
                query = str(call["args"].get("query", "")).strip()
                if call["name"] != SEARCH_TOOL_NAME:
                    content = f"Unknown tool {call['name']!r}; only {SEARCH_TOOL_NAME} is available."
                elif not query:
                    content = "A non-empty query is required."
                elif calls >= budget:
                    content = "Search budget exhausted. Finish with the evidence you already have."
                else:
                    calls += 1
                    try:
                        rows = mcp.search(query, server=outcome.mcp_server, tool=outcome.mcp_tool)
                    except Exception as exc:  # retrieval failures are task outcomes, not crashes
                        error = redact_mcp_error(exc)
                        errors.append(error)
                        content = f"Search failed: {error}"
                    else:
                        fresh, repeated = [], []
                        for row in rows:
                            key = normalize_url(row.get("url", ""))
                            if key in seen_urls:
                                repeated.append(str(row.get("url", "")))
                            else:
                                seen_urls.add(key)
                                fresh.append(row)
                        sources.extend({**row, "_query": query} for row in fresh)
                        content = _format_search_results(query, fresh, repeated)
                replies.append(ToolMessage(content=content, tool_call_id=call["id"]))
            return {"messages": replies, "search_calls": calls, "sources": sources, "errors": errors}

        def compress_research(state: ResearcherState) -> ResearcherState:
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
                    prompt_version=COMPRESS_PROMPT_VERSION,
                    run_id=run_id,
                    thread_id=thread_id,
                )
            except Exception as exc:
                return {"findings": None, "errors": [f"Compression failed: {exc}"]}
            return {"findings": findings}

        def route_after_llm(state: ResearcherState) -> str:
            last = state["messages"][-1]
            wants_search = isinstance(last, AIMessage) and bool(last.tool_calls)
            if (
                wants_search
                and state.get("search_calls", 0) < budget
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
