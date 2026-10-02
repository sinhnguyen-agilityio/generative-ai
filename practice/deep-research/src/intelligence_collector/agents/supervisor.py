"""Research Supervisor: delegate planned research questions to research sub-agents.

One sub-agent per planned todo, up to ``max_concurrent_researchers`` running at once. Sub-agents
only search and compress, so they run in parallel; VFS and vector-index writes happen afterwards
in plan order on the calling thread.
"""

from __future__ import annotations

import asyncio

from intelligence_collector.agents.researcher import ResearchAgent, ResearchOutcome
from intelligence_collector.agents.worker import already_researched, failed_outcome, persist_research
from intelligence_collector.contracts import ProgressAssessment, ResearchTodo, TodoStatus
from intelligence_collector.tools.mcp import redact_mcp_error
from intelligence_collector.tools.think_tool import think_tool
from intelligence_collector.tools.url_tracking import UrlClaims, UrlRegistry

DEFAULT_MAX_CONCURRENT_RESEARCHERS = 3
MAX_CONCURRENT_RESEARCHERS = 5


def assess(todos, artifacts, indexes) -> ProgressAssessment:
    """ResearchComplete gate: every todo completed, artifacts persisted, data indexed."""
    return think_tool(todos, len(artifacts), len(indexes))


class ResearchSupervisor:
    """Delegates research questions to parallel sub-agents (bounded concurrency), skips todos
    already researched, and persists each outcome.
    """

    def __init__(
        self,
        researcher: ResearchAgent,
        vfs,
        index,
        *,
        max_concurrent_researchers: int = DEFAULT_MAX_CONCURRENT_RESEARCHERS,
    ):
        self.researcher = researcher
        self.vfs = vfs
        self.index = index
        self.max_concurrent_researchers = min(
            max(1, max_concurrent_researchers), MAX_CONCURRENT_RESEARCHERS
        )

    def delegate(self, run_id: str, todos: list[ResearchTodo], *, thread_id: str = ""):
        """Research every todo; returns ``(todo, refs, chunks)`` per todo in plan order."""
        pending = [todo for todo in todos if todo.status == TodoStatus.PLANNED
                   and not already_researched(self.vfs, run_id, todo)]
        # One run-level URL registry, so a page found by several sub-agents is captured once.
        registry = UrlRegistry.load(self.vfs, run_id)
        # Shared by the parallel sub-agents while they search, so a page one task already
        # retrieved (now or earlier in the run) is not pulled into another task at all.
        claims = UrlClaims(registry.owners())
        outcomes = self._run_sub_agents(run_id, pending, thread_id, claims) if pending else {}

        results = []
        for todo in todos:
            if todo.todo_id not in outcomes:
                if already_researched(self.vfs, run_id, todo):
                    todo.status = TodoStatus.COMPLETED
                results.append((todo, [], []))
            else:
                results.append(persist_research(
                    self.vfs, self.index, run_id, todo, outcomes[todo.todo_id], registry
                ))
        return results

    def _run_sub_agents(
        self, run_id: str, todos: list[ResearchTodo], thread_id: str, claims: UrlClaims
    ) -> dict[str, ResearchOutcome]:
        """Synchronous entry point (called from the sync ``researching`` graph node): drives the
        async fan-out below to completion on a short-lived event loop, since nothing else in the
        call chain is async."""
        return asyncio.run(self._gather_sub_agents(run_id, todos, thread_id, claims))

    async def _gather_sub_agents(
        self, run_id: str, todos: list[ResearchTodo], thread_id: str, claims: UrlClaims
    ) -> dict[str, ResearchOutcome]:
        # Each sub-agent's actual work (LLM calls, MCP search) is still synchronous, so
        # asyncio.to_thread runs it in a worker thread -- same real parallelism as before, capped
        # by the semaphore since asyncio.to_thread's own thread pool has no such limit. Unlike a
        # bare ThreadPoolExecutor, asyncio.to_thread propagates the calling contextvars.Context
        # into that thread automatically, so Langfuse/OpenTelemetry's "current span" and
        # LangChain's runnable config still reach each sub-agent's LLM call traces without any
        # manual context handling here.
        semaphore = asyncio.Semaphore(self.max_concurrent_researchers)

        async def bounded(todo: ResearchTodo) -> tuple[str, ResearchOutcome]:
            """Research one todo while holding a concurrency slot."""
            async with semaphore:
                outcome = await asyncio.to_thread(
                    self._research_one, run_id, todo, thread_id, claims
                )
                return todo.todo_id, outcome

        results = await asyncio.gather(*(bounded(todo) for todo in todos))
        return dict(results)

    def _research_one(
        self, run_id: str, todo: ResearchTodo, thread_id: str, claims: UrlClaims
    ) -> ResearchOutcome:
        # One failing sub-agent must not cancel its siblings; the failure becomes its outcome.
        try:
            return self.researcher.research(
                todo.research_question + (
                    "\nResearch only this missing evidence: " + todo.research_instruction
                    if todo.research_instruction else ""
                ),
                run_id=run_id,
                thread_id=thread_id,
                task_id=todo.todo_id,
                claims=claims,
            )
        except Exception as exc:
            return failed_outcome(todo, f"Research sub-agent failed: {redact_mcp_error(exc)}")

    def assess(self, todos, artifacts, index_refs) -> ProgressAssessment:
        """Decide whether research is complete enough to synthesize (see ``tools/think_tool.py``).
        """
        return assess(todos, artifacts, index_refs)
