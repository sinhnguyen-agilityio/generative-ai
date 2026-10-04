"""Research Supervisor: delegate planned research questions to research sub-agents.

One sub-agent per planned todo, up to ``max_concurrent_researchers`` running at once. Sub-agents
only search and compress, so they run in parallel; VFS and vector-index writes happen as each
outcome completes on the calling thread.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

from intelligence_collector.agents.researcher import ResearchAgent, ResearchOutcome
from intelligence_collector.agents.worker import (
    already_researched,
    failed_outcome,
    persist_research,
)
from intelligence_collector.agents.worker import restore_research
from intelligence_collector.agents.worker import research_prefix
from intelligence_collector.contracts import ResearchFindings
from intelligence_collector.contracts import ResearchTodo, TodoStatus
from intelligence_collector.observability.tracing import traced
from intelligence_collector.tools.mcp import redact_mcp_error
from intelligence_collector.tools.url_tracking import UrlClaims, UrlRegistry

DEFAULT_MAX_CONCURRENT_RESEARCHERS = 3
MAX_CONCURRENT_RESEARCHERS = 5


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
        pending = [
            todo
            for todo in todos
            if todo.status == TodoStatus.PLANNED and not already_researched(self.vfs, run_id, todo)
        ]
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
                    results.append(restore_research(self.vfs, run_id, todo))
                else:
                    results.append((todo, [], []))
            else:
                results.append(outcomes[todo.todo_id])
        return results

    def _run_sub_agents(
        self, run_id: str, todos: list[ResearchTodo], thread_id: str, claims: UrlClaims
    ) -> dict:
        """Synchronous entry point (called from the sync ``researching`` graph node): drives the
        async fan-out below to completion on a short-lived event loop, since nothing else in the
        call chain is async."""
        return asyncio.run(self._gather_sub_agents(run_id, todos, thread_id, claims))

    async def _gather_sub_agents(
        self, run_id: str, todos: list[ResearchTodo], thread_id: str, claims: UrlClaims
    ) -> dict:
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

        tasks = [asyncio.create_task(bounded(todo)) for todo in todos]
        registry = UrlRegistry.load(self.vfs, run_id)
        by_id = {todo.todo_id: todo for todo in todos}
        results = {}
        errors = []
        for finished in asyncio.as_completed(tasks):
            todo_id, outcome = await finished
            try:
                with traced("research.persist", metadata={"todo_id": todo_id}) as observation:
                    payload = asdict(outcome)
                    payload["findings"] = (
                        outcome.findings.model_dump(mode="json") if outcome.findings else None
                    )
                    self.vfs.write_json(
                        run_id,
                        f"{research_prefix(by_id[todo_id])}/retrieved_outcome.json",
                        payload,
                        "retrieved_outcome",
                        task_id=todo_id,
                    )
                    results[todo_id] = persist_research(
                        self.vfs, self.index, run_id, by_id[todo_id], outcome, registry
                    )
                    todo, refs, chunks = results[todo_id]
                    observation.output = {
                        "status": todo.status.value,
                        "artifacts": len(refs),
                        "indexed_chunks": len(chunks),
                    }
            except Exception as exc:
                # Finish persisting siblings even if one local index/storage operation failed.
                errors.append(exc)
                # Drop uncommitted URL registrations from the failed persistence attempt.
                registry = UrlRegistry.load(self.vfs, run_id)
        if errors:
            raise RuntimeError(
                "Research persistence failed; completed sibling artifacts are recoverable"
            ) from errors[0]
        return results

    def _research_one(
        self, run_id: str, todo: ResearchTodo, thread_id: str, claims: UrlClaims
    ) -> ResearchOutcome:
        # One failing sub-agent must not cancel its siblings; the failure becomes its outcome.
        try:
            try:
                payload = json.loads(
                    self.vfs.read(run_id, f"{research_prefix(todo)}/retrieved_outcome.json")
                )
            except FileNotFoundError:
                payload = None
            if payload is not None:
                if payload.get("findings") is not None:
                    payload["findings"] = ResearchFindings.model_validate(payload["findings"])
                return ResearchOutcome(**payload)
            return self.researcher.research(
                todo.research_question
                + (
                    "\nResearch only this missing evidence: " + todo.research_instruction
                    if todo.research_instruction
                    else ""
                ),
                run_id=run_id,
                thread_id=thread_id,
                task_id=todo.todo_id,
                claims=claims,
            )
        except Exception as exc:
            return failed_outcome(todo, f"Research sub-agent failed: {redact_mcp_error(exc)}")
