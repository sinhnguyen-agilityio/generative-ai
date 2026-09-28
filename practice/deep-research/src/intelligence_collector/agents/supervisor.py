"""Research Supervisor: delegate planned research questions to research sub-agents.

One sub-agent per planned todo, up to ``max_concurrent_researchers`` running at once. Sub-agents
only search and compress, so they run in parallel; VFS and vector-index writes happen afterwards
in plan order on the calling thread.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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
        pending = [todo for todo in todos if not already_researched(self.vfs, run_id, todo)]
        # One run-level URL registry, so a page found by several sub-agents is captured once.
        registry = UrlRegistry.load(self.vfs, run_id)
        # Shared by the parallel sub-agents while they search, so a page one task already
        # retrieved (now or earlier in the run) is not pulled into another task at all.
        claims = UrlClaims(registry.owners())
        outcomes = self._run_sub_agents(run_id, pending, thread_id, claims) if pending else {}

        results = []
        for todo in todos:
            if todo.todo_id not in outcomes:
                todo.status = TodoStatus.COMPLETED  # researched in an earlier invocation
                results.append((todo, [], []))
            else:
                results.append(persist_research(
                    self.vfs, self.index, run_id, todo, outcomes[todo.todo_id], registry
                ))
        return results

    def _run_sub_agents(
        self, run_id: str, todos: list[ResearchTodo], thread_id: str, claims: UrlClaims
    ) -> dict[str, ResearchOutcome]:
        workers = min(len(todos), self.max_concurrent_researchers)
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="research") as pool:
            futures = {
                todo.todo_id: pool.submit(self._research_one, run_id, todo, thread_id, claims)
                for todo in todos
            }
            return {todo_id: future.result() for todo_id, future in futures.items()}

    def _research_one(
        self, run_id: str, todo: ResearchTodo, thread_id: str, claims: UrlClaims
    ) -> ResearchOutcome:
        # One failing sub-agent must not cancel its siblings; the failure becomes its outcome.
        try:
            return self.researcher.research(
                todo.research_question,
                run_id=run_id,
                thread_id=thread_id,
                task_id=todo.todo_id,
                claims=claims,
            )
        except Exception as exc:
            return failed_outcome(todo, f"Research sub-agent failed: {redact_mcp_error(exc)}")

    def assess(self, todos, artifacts, index_refs) -> ProgressAssessment:
        return assess(todos, artifacts, index_refs)
