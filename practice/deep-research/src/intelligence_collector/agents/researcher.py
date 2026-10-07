"""Research sub-agent (the *searcher*): an LLM tool-calling loop over ``tavily_search`` and
``think_tool`` for one research task.

Each searcher investigates exactly one task in an isolated context::

    START -> llm_call --(tool calls, no stop condition)--> tool_node --> llm_call
                     \\--(final answer or a stop condition)--> END

* ``llm_call``  LLM decision: search, reflect with ``think_tool``, or give a final note.
* ``tool_node`` Runs each search through the MCP boundary, keeps each fetched page's most relevant
                verbatim passages (``passage_selector.py``), and has every result accepted or
                rejected with a reason (``result_evaluation.py``). Only accepted results count
                toward the page budget and become evidence. Repeated queries are refused.
* Stop conditions (recorded as ``StopReason``): every evidence point covered, search budget
  spent, page budget full, two searches in a row with no accepted result, two failed searches
  in a row (a failed result evaluation counts), or the model ending the search itself.

``search`` returns the gathered sources and decisions; the task's short report is written from
them by ``searcher_report.py`` (see ``search_task.py``). The LLM only ever reads the selected
passages; each page's full text travels in ``sources`` to ``worker.py``,
which stores it in the VFS and indexes its chunks. Reflections stay in the message history.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Any, TypedDict, cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from intelligence_collector.ports import ModelRouterPort
from intelligence_collector.agents.passage_selector import PassageSelector
from intelligence_collector.agents.prompts import RESEARCHER_TEMPLATE
from intelligence_collector.agents.result_evaluation import evaluate_results
from intelligence_collector.agents.tools import (
    SEARCH_TOOL,
    SEARCH_TOOL_NAME,
    THINK_TOOL,
    THINK_TOOL_NAME,
    think,
)
from intelligence_collector.contracts import (
    FailureKind,
    RejectReason,
    ResearchFindings,
    ResearchTodo,
    ResultEvaluation,
    StopReason,
)
from intelligence_collector.observability.tracing import traced
from intelligence_collector.models.redaction import redact_error
from intelligence_collector.tools.mcp import MCPResearchPort
from intelligence_collector.tools.url_tracking import UrlClaims, normalize_url

HARD_SEARCH_LIMIT = 5
# Distinct accepted pages one searcher may keep for its task, across all of its searches — not
# per search call.
DEFAULT_MAX_SOURCES = 5
HARD_SOURCE_LIMIT = 10
# Passages are bounded by the selector (three chunks per page); this only guards against snippets
# that run long.
TOOL_RESULT_CHARS = 3200
# Consecutive searches without an accepted result, or failed, that end the search (FR-007).
MAX_EMPTY_SEARCHES = 2
MAX_FAILED_SEARCHES = 2
_WORDS = re.compile(r"\w+")


def normalized_query(query: str) -> str:
    """Identity of a query for the repeat check: case, spacing and word order ignored."""
    return " ".join(sorted(_WORDS.findall(query.casefold())))


class ResearcherState(TypedDict, total=False):
    """State of one searcher's own LangGraph loop."""

    research_question: str
    messages: Annotated[list[BaseMessage], add_messages]
    llm_calls: int
    search_calls: int
    sources: Annotated[list[dict[str, Any]], operator.add]
    errors: Annotated[list[str], operator.add]
    evaluations: Annotated[list[dict[str, Any]], operator.add]
    corrections: Annotated[list[str], operator.add]
    queries: Annotated[list[str], operator.add]
    covered: Annotated[list[str], operator.add]
    empty_streak: int
    failure_streak: int
    owned: int


@dataclass
class ResearchOutcome:
    """Everything one searcher gathered for its task."""

    question: str
    search_budget: int
    mcp_server: str
    mcp_tool: str
    search_calls: int = 0
    sources: list[dict[str, Any]] = field(default_factory=list)
    findings: ResearchFindings | None = None
    errors: list[str] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    covered_points: list[str] = field(default_factory=list)
    stop_reason: str = StopReason.MODEL_FINISHED.value
    failure_kind: str | None = None

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


def _format_search_results(
    query: str,
    rows: list[dict[str, Any]],
    evaluations: list[ResultEvaluation],
    uncovered: list[str],
) -> str:
    """The tool reply: accepted pages with their passages, rejected ones with the reason, and the
    evidence points still uncovered."""
    if not rows:
        blocks = [f'No new results for "{query}".']
    else:
        blocks = [f'Search results for "{query}":']
        for number, (row, evaluation) in enumerate(zip(rows, evaluations), start=1):
            if evaluation.decision == "rejected":
                blocks.append(
                    f"[{number}] REJECTED ({evaluation.reason.value}) "
                    f"{row.get('title') or 'Untitled'}\nURL: {row.get('url', '')}"
                )
                continue
            linked = (
                " (already retrieved by another research task: linked, does not use your "
                "source budget)"
                if row.get("_covered_by")
                else ""
            )
            snippet = " (snippet only: no readable page)" if evaluation.snippet_only else ""
            blocks.append(
                f"[{number}] ACCEPTED{linked}{snippet} {row.get('title') or 'Untitled'}\n"
                f"URL: {row.get('url', '')}\n"
                f"{str(row.get('content', ''))[:TOOL_RESULT_CHARS]}"
            )
    if uncovered:
        blocks.append(
            "Evidence points still uncovered (target these in your next query):\n"
            + "\n".join(f"- {text}" for text in uncovered)
        )
    else:
        blocks.append("Every evidence point of this task is covered.")
    return "\n\n".join(blocks)


def stop_reason(
    state: ResearcherState,
    *,
    point_ids: set[str],
    search_budget: int,
    source_budget: int,
    max_llm_calls: int,
) -> StopReason | None:
    """The first stop condition that holds after the latest LLM turn, or None to continue."""
    if point_ids and point_ids <= set(state.get("covered", [])):
        return StopReason.ALL_POINTS_COVERED
    if state.get("failure_streak", 0) >= MAX_FAILED_SEARCHES:
        return StopReason.SEARCH_FAILURES
    if state.get("empty_streak", 0) >= MAX_EMPTY_SEARCHES:
        return StopReason.NO_PROGRESS
    if state.get("search_calls", 0) >= search_budget:
        return StopReason.SEARCH_BUDGET_SPENT
    if state.get("owned", 0) >= source_budget:
        return StopReason.PAGE_BUDGET_FULL
    last = state["messages"][-1] if state.get("messages") else None
    wants_tool = isinstance(last, AIMessage) and bool(last.tool_calls)
    if not wants_tool or state.get("llm_calls", 0) >= max_llm_calls:
        return StopReason.MODEL_FINISHED
    return None


class ResearchAgent:
    """Runs one isolated searcher per research task."""

    def __init__(
        self,
        router: ModelRouterPort,
        mcp_factory: Callable[[], MCPResearchPort],
        *,
        max_search_calls: int = HARD_SEARCH_LIMIT,
        max_sources: int = DEFAULT_MAX_SOURCES,
        passage_selector: PassageSelector | None = None,
        evaluator_router: ModelRouterPort | None = None,
        report_router: ModelRouterPort | None = None,
    ):
        self.router = router
        # Result evaluation and the short report run on the same (standard) tier unless other
        # routers are given.
        self.evaluator_router = evaluator_router or router
        self.report_router = report_router or router
        self.mcp_factory = mcp_factory
        self.max_search_calls = min(max(1, max_search_calls), HARD_SEARCH_LIMIT)
        self.max_sources = min(max(1, max_sources), HARD_SOURCE_LIMIT)
        self.passage_selector = passage_selector or PassageSelector()

    def search(
        self,
        question: str,
        *,
        todo: ResearchTodo | None = None,
        run_id: str = "",
        thread_id: str = "",
        task_id: str = "",
        claims: UrlClaims | None = None,
        known_passages: str = "",
        force_first_search: bool = True,
    ) -> ResearchOutcome:
        """Run one searcher for `question` (the task's evidence points come from `todo`) within
        the search and page budgets. `claims` is shared by parallel searchers so each page is
        captured by one task only. `known_passages` are passages the run already stored that may
        answer some points (FR-013b); without `force_first_search` the searcher may finish
        without searching when they suffice."""
        todo = todo or ResearchTodo(research_question=question, todo_id=task_id or "task")
        task_id = task_id or todo.todo_id
        # A fresh MCP client per searcher keeps search budgets and request logs per task.
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
            todo,
            run_id=run_id,
            thread_id=thread_id,
            task_id=task_id,
            claims=claims,
            force_first_search=force_first_search,
        )
        points = "\n".join(f"- {p.point_id}: {p.text}" for p in todo.evidence_points)
        opening = f"Research question:\n{question}"
        if points:
            opening += f"\n\nEvidence points this task must cover:\n{points}"
        if known_passages:
            opening += (
                "\n\nPassages the run already stored that may answer some points (search only "
                f"for what they leave open):\n{known_passages}"
            )
        final = graph.invoke(
            {
                "research_question": question,
                "messages": [HumanMessage(content=opening)],
                "llm_calls": 0,
                "search_calls": 0,
                "empty_streak": 0,
                "failure_streak": 0,
                "owned": 0,
            },
            config={
                "run_name": f"research_agent · {task_id}" if task_id else "research_agent",
                "metadata": {"todo_id": task_id, "run_id": run_id},
            },
        )
        outcome.search_calls = final.get("search_calls", 0)
        outcome.sources = final.get("sources", [])
        outcome.errors = final.get("errors", [])
        outcome.evaluations = final.get("evaluations", [])
        outcome.corrections = final.get("corrections", [])
        outcome.queries = final.get("queries", [])
        outcome.covered_points = list(dict.fromkeys(final.get("covered", [])))
        reason = stop_reason(
            final,
            point_ids={p.point_id for p in todo.evidence_points},
            search_budget=budget,
            source_budget=self.max_sources,
            max_llm_calls=2 * budget + 2,
        )
        outcome.stop_reason = (reason or StopReason.MODEL_FINISHED).value
        if not outcome.sources:
            outcome.failure_kind = (
                FailureKind.TECHNICAL if outcome.errors else FailureKind.CONTENT
            ).value
        return outcome

    def _build_graph(
        self,
        mcp: MCPResearchPort,
        outcome: ResearchOutcome,
        todo: ResearchTodo,
        *,
        run_id,
        thread_id,
        task_id="",
        claims: UrlClaims | None = None,
        force_first_search: bool = True,
    ):
        budget = outcome.search_budget
        source_budget = self.max_sources
        # One reflection per search, plus a couple of spare turns (e.g. a malformed tool call)
        # without looping forever.
        max_llm_calls = 2 * budget + 2
        point_ids = {p.point_id for p in todo.evidence_points}
        point_text = {p.point_id: p.text for p in todo.evidence_points}
        # URLs this searcher has already evaluated; repeats are rejected as duplicates.
        seen_urls: set[str] = set()
        used_queries: set[str] = set()

        def llm_call(state: ResearcherState) -> ResearcherState:
            """Ask the searcher LLM for its next step: search, reflect, or give a final note."""
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
                tool_choice=SEARCH_TOOL_NAME
                if force_first_search and not state.get("search_calls")
                else None,
            )
            return {"messages": [response], "llm_calls": state.get("llm_calls", 0) + 1}

        def read_pages(question: str, query: str, rows: list[dict[str, Any]]):
            """Select the relevant passages of every page this task could keep; pages another
            task already owns keep their snippet and are linked, not re-read."""
            readable = [row for row in rows if not row.get("_covered_by")]
            read = iter(self.passage_selector.read_pages(f"{question}\n{query}", readable))
            return [row if row.get("_covered_by") else next(read) for row in rows]

        def search(state: ResearcherState, query: str) -> dict[str, Any]:
            """One search: fetch, read, evaluate, keep accepted rows. Returns the tool reply and
            the state changes."""
            question = state["research_question"]
            remaining = source_budget - state.get("owned", 0)
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
                error = redact_error(exc)
                return {"reply": f"Search failed: {error}", "errors": [error], "failed": True}
            # Mark pages another task already owns before reading, so they are not read again.
            marked = []
            for row in rows:
                url = str(row.get("url", ""))
                owner = claims.owner(url) if claims is not None else None
                if owner is not None and owner != task_id and normalize_url(url) not in seen_urls:
                    marked.append({**row, "raw_content": "", "_covered_by": owner})
                else:
                    marked.append(row)
            fresh_positions = [
                i for i, row in enumerate(marked) if normalize_url(str(row.get("url", ""))) not in seen_urls
            ]
            fresh = read_pages(question, query, [marked[i] for i in fresh_positions])
            for position, row in zip(fresh_positions, fresh):
                marked[position] = row
            evaluation = evaluate_results(
                marked,
                todo=todo,
                question=question,
                query=query,
                search_index=state.get("search_calls", 0) + 1,
                seen_urls=seen_urls,
                router=self.evaluator_router,
                run_id=run_id,
                thread_id=thread_id,
            )
            for row in marked:
                seen_urls.add(normalize_url(str(row.get("url", ""))))
            kept: list[dict[str, Any]] = []
            evaluations: list[ResultEvaluation] = []
            owned = state.get("owned", 0)
            for row, decision in zip(marked, evaluation.evaluations):
                if decision.decision == "accepted":
                    url = str(row.get("url", ""))
                    if row.get("_covered_by"):
                        decision = decision.model_copy(
                            update={"linked_to_task": row["_covered_by"], "snippet_only": False}
                        )
                        kept.append(row)
                    elif owned >= source_budget:
                        evaluation.corrections.append(
                            f"{url}: accepted but the page budget is full; not kept"
                        )
                        decision = ResultEvaluation(
                            url=url,
                            title=decision.title,
                            query=query,
                            search_index=decision.search_index,
                            decision="rejected",
                            reason=RejectReason.OUTSIDE_SCOPE,
                        )
                    else:
                        owner = claims.claim(url, task_id) if claims is not None else task_id
                        if owner != task_id:
                            row = {**row, "raw_content": "", "_covered_by": owner}
                            decision = decision.model_copy(
                                update={"linked_to_task": owner, "snippet_only": False}
                            )
                        else:
                            owned += 1
                        kept.append(row)
                evaluations.append(decision)
            covered = [c for e in evaluations if e.decision == "accepted" for c in e.covers]
            uncovered_ids = sorted(point_ids - set(state.get("covered", [])) - set(covered))
            reply = _format_search_results(
                query, marked, evaluations, [point_text[i] for i in uncovered_ids]
            )
            if owned >= source_budget:
                reply += (
                    f"\n\nSource budget reached ({source_budget} pages). "
                    "Finish with the evidence you already have."
                )
            accepted = sum(e.decision == "accepted" for e in evaluations)
            return {
                "reply": reply,
                "sources": [{**row, "_query": query} for row in kept],
                "evaluations": [e.model_dump(mode="json") for e in evaluations],
                "corrections": list(evaluation.corrections),
                "errors": [evaluation.error] if evaluation.failed else [],
                "covered": covered,
                "owned": owned,
                "failed": evaluation.failed,
                "accepted": accepted,
            }

        def tool_node(state: ResearcherState) -> ResearcherState:
            """Run each requested search or reflection within the budgets."""
            update: dict[str, Any] = {
                "search_calls": state.get("search_calls", 0),
                "owned": state.get("owned", 0),
                "empty_streak": state.get("empty_streak", 0),
                "failure_streak": state.get("failure_streak", 0),
                "messages": [],
                "sources": [],
                "errors": [],
                "evaluations": [],
                "corrections": [],
                "queries": [],
                "covered": [],
            }
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
                elif normalized_query(query) in used_queries:
                    content = "Refused: repeats an earlier query of this task."
                elif update["search_calls"] >= budget:
                    content = "Search budget exhausted. Finish with the evidence you already have."
                elif update["owned"] >= source_budget:
                    content = (
                        f"Source budget reached ({source_budget} pages). "
                        "Finish with the evidence you already have."
                    )
                else:
                    used_queries.add(normalized_query(query))
                    update["search_calls"] += 1
                    current = {**state, **{k: update[k] for k in ("search_calls", "owned")},
                               "covered": [*state.get("covered", []), *update["covered"]]}
                    result = search(current, query)
                    content = result["reply"]
                    update["queries"].append(query)
                    update["errors"].extend(result.get("errors", []))
                    if result.get("failed"):
                        update["failure_streak"] += 1
                    else:
                        update["failure_streak"] = 0
                    if "evaluations" in result:
                        update["sources"].extend(result["sources"])
                        update["evaluations"].extend(result["evaluations"])
                        update["corrections"].extend(result["corrections"])
                        update["covered"].extend(result["covered"])
                        update["owned"] = result["owned"]
                        update["empty_streak"] = (
                            0 if result["accepted"] else update["empty_streak"] + 1
                        )
                update["messages"].append(ToolMessage(content=content, tool_call_id=call["id"]))
            return update

        def route_after_llm(state: ResearcherState) -> str:
            """Continue while the LLM asks for a tool and no stop condition holds."""
            reason = stop_reason(
                state,
                point_ids=point_ids,
                search_budget=budget,
                source_budget=source_budget,
                max_llm_calls=max_llm_calls,
            )
            return END if reason is not None else "tool_node"

        graph = StateGraph(ResearcherState)
        graph.add_node("llm_call", llm_call)
        graph.add_node("tool_node", tool_node)
        graph.add_edge(START, "llm_call")
        graph.add_conditional_edges("llm_call", route_after_llm, ["tool_node", END])
        graph.add_edge("tool_node", "llm_call")
        return graph.compile()
