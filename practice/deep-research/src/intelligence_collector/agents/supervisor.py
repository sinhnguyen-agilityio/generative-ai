"""Research Supervisor: an LLM that delegates the approved research tasks to sub-agents.

The supervisor model decides every step with four tools::

    think_tool        reflect and plan (before delegating, after each batch)
    ConductResearch   research one approved task (by todo_id) with a dedicated sub-agent;
                      several calls in one response run in parallel
    ResearchComplete  stop; the report writer takes over
    ProposeReplan     the approved plan cannot be answered as written: ask a human for a new one

Code keeps the model inside the approved plan: unknown todo ids are refused, each task gets at
most ``max_attempts_per_task`` research rounds, ResearchComplete is refused while a task was never
researched, and ``max_turns`` bounds the loop. A task the model never delegated before the turn
limit is researched anyway, so no approved task is skipped.

A second round for the same task is a follow-up: it researches only the missing evidence the
model names, and its findings are appended to the earlier ones. Sub-agents only search and
compress, so they run in parallel (``max_concurrent_researchers`` at once); VFS and vector-index
writes happen as each outcome completes on the calling thread.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

from intelligence_collector.ports import ArtifactStore, ModelRouterPort, VectorIndex
from intelligence_collector.agents.prompts import SUPERVISOR_REPLAN_TOOL, SUPERVISOR_TEMPLATE
from intelligence_collector.agents.researcher import ResearchAgent, ResearchOutcome
from intelligence_collector.agents.tools import (
    CONDUCT_RESEARCH_TOOL,
    CONDUCT_RESEARCH_TOOL_NAME,
    PROPOSE_REPLAN_TOOL,
    PROPOSE_REPLAN_TOOL_NAME,
    RESEARCH_COMPLETE_TOOL,
    RESEARCH_COMPLETE_TOOL_NAME,
    THINK_TOOL,
    THINK_TOOL_NAME,
    think,
)
from intelligence_collector.agents.worker import (
    _evidence_path,
    already_researched,
    failed_outcome,
    persist_research,
    research_prefix,
    restore_research,
)
from intelligence_collector.contracts import ResearchFindings, ResearchTodo, TodoStatus
from intelligence_collector.observability.tracing import traced
from intelligence_collector.models.redaction import redact_error
from intelligence_collector.tools.url_tracking import UrlClaims, UrlRegistry

DEFAULT_MAX_CONCURRENT_RESEARCHERS = 3
MAX_CONCURRENT_RESEARCHERS = 5
DEFAULT_MAX_SUPERVISOR_TURNS = 8
DEFAULT_MAX_ATTEMPTS_PER_TASK = 2
SUPERVISOR_LOG_PATH = "research/supervisor_log.json"


# Set by a UI that shows decisions live. It is called on the thread running the graph node, which
# for a ``stream_mode="values"`` run is the caller's thread (LangGraph's "custom" stream mode
# would move nodes onto a thread pool, out of reach of the UI framework's session).
decision_listener: ContextVar[Callable[[dict], None] | None] = ContextVar(
    "supervisor_decision_listener", default=None
)


class DecisionLog:
    """What the supervisor decided, for people watching the run: appended to
    ``research/supervisor_log.json`` in the VFS and passed to the ``decision_listener``."""

    def __init__(self, vfs: ArtifactStore, run_id: str):
        self.vfs, self.run_id = vfs, run_id
        try:
            self.entries: list[dict] = json.loads(vfs.read(run_id, SUPERVISOR_LOG_PATH))
        except FileNotFoundError:
            self.entries = []

    def add(self, kind: str, **fields) -> None:
        entry = {"kind": kind, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        entry.update(fields)
        self.entries.append(entry)
        self.vfs.write_json(self.run_id, SUPERVISOR_LOG_PATH, self.entries, "supervisor_log")
        listener = decision_listener.get()
        if listener is not None:
            try:
                listener(entry)
            except Exception:  # a display problem must never stop research
                pass


def read_supervisor_log(vfs: ArtifactStore, run_id: str) -> list[dict]:
    """The run's supervisor decisions, oldest first; empty before any research."""
    try:
        return json.loads(vfs.read(run_id, SUPERVISOR_LOG_PATH))
    except FileNotFoundError:
        return []


@dataclass
class ReplanProposal:
    """The supervisor's request for a replacement plan, for human approval."""

    reason: str
    todo_ids: list[str]


@dataclass
class ResearchJob:
    """One ConductResearch call: the task, the supervisor's topic, whether it follows up an
    earlier round of the same task, and the tool call it answers ("" when code started it)."""

    todo: ResearchTodo
    topic: str
    follow_up: bool = False
    call_id: str = ""


class Delegation(list):
    """``(todo, refs, chunks)`` per todo in plan order, plus the supervisor's replan proposal."""

    replan: ReplanProposal | None = None


class ResearchSupervisor:
    """LLM supervisor over parallel research sub-agents; persists each outcome."""

    def __init__(
        self,
        researcher: ResearchAgent,
        vfs: ArtifactStore,
        index: VectorIndex | None,
        *,
        router: ModelRouterPort | None = None,
        max_concurrent_researchers: int = DEFAULT_MAX_CONCURRENT_RESEARCHERS,
        max_turns: int = DEFAULT_MAX_SUPERVISOR_TURNS,
        max_attempts_per_task: int = DEFAULT_MAX_ATTEMPTS_PER_TASK,
    ):
        self.researcher = researcher
        self.vfs = vfs
        self.index = index
        self.router = router or getattr(researcher, "router", None)
        self.max_concurrent_researchers = min(
            max(1, max_concurrent_researchers), MAX_CONCURRENT_RESEARCHERS
        )
        self.max_turns = max(1, max_turns)
        self.max_attempts_per_task = max(1, max_attempts_per_task)
        # Set by the graph builder: replacement plans need a Planner and a plan approval step.
        self.replan_enabled = False

    # ----- public entry point -----

    def delegate(
        self,
        run_id: str,
        todos: list[ResearchTodo],
        *,
        thread_id: str = "",
        brief: str = "",
        allow_replan: bool = False,
    ) -> Delegation:
        """Research every planned todo under the supervisor model; returns ``(todo, refs,
        chunks)`` per todo in plan order. Todos researched earlier are restored, not repeated."""
        pending = [
            todo
            for todo in todos
            if todo.status == TodoStatus.PLANNED and not already_researched(self.vfs, run_id, todo)
        ]
        delegation = Delegation()
        outcomes: dict[str, tuple] = {}
        if pending:
            outcomes, delegation.replan = self._supervise(
                run_id,
                pending,
                thread_id=thread_id,
                brief=brief,
                allow_replan=allow_replan and self.replan_enabled,
            )
        for todo in todos:
            if todo.todo_id in outcomes:
                delegation.append(outcomes[todo.todo_id])
            elif already_researched(self.vfs, run_id, todo):
                delegation.append(restore_research(self.vfs, run_id, todo))
            else:
                delegation.append((todo, [], []))
        return delegation

    # ----- supervisor loop -----

    def _supervise(self, run_id, pending, *, thread_id, brief, allow_replan):
        """Run the supervisor model until ResearchComplete, a replan proposal or the turn limit."""
        if self.router is None:
            raise ValueError("The research supervisor needs a model router")
        # One run-level URL registry, so a page found by several sub-agents is captured once.
        registry = UrlRegistry.load(self.vfs, run_id)
        # Shared by the parallel sub-agents while they search, so a page one task already
        # retrieved (now or earlier in the run) is not pulled into another task at all.
        claims = UrlClaims(registry.owners())
        by_id = {todo.todo_id: todo for todo in pending}
        rounds: Counter[str] = Counter()
        results: dict[str, tuple] = {}
        replan: ReplanProposal | None = None
        tools = [THINK_TOOL, CONDUCT_RESEARCH_TOOL, RESEARCH_COMPLETE_TOOL]
        if allow_replan:
            tools.append(PROPOSE_REPLAN_TOOL)
        messages: list[BaseMessage] = [HumanMessage(content=self._task_list(pending, brief))]
        log = DecisionLog(self.vfs, run_id)

        for _ in range(self.max_turns):
            response = self.router.invoke_with_tools(
                "supervisor",
                SUPERVISOR_TEMPLATE.format_messages(
                    messages=messages,
                    date=date.today().isoformat(),
                    replan_tool=SUPERVISOR_REPLAN_TOOL if allow_replan else "",
                    max_concurrent_research_units=self.max_concurrent_researchers,
                    max_attempts_per_task=self.max_attempts_per_task,
                    max_turns=self.max_turns,
                ),
                tools,
                run_id=run_id,
                thread_id=thread_id,
            )
            messages.append(response)
            if not response.tool_calls:
                unresearched = self._unresearched(pending, rounds)
                if not unresearched:
                    break
                messages.append(HumanMessage(content=self._incomplete(unresearched)))
                log.add("refused", text=self._incomplete(unresearched))
                continue
            replies, jobs, finished, replan = self._decide(
                response.tool_calls, by_id, rounds, pending, allow_replan, log
            )
            if jobs:
                replies.update(self._research_batch(run_id, jobs, thread_id, claims, results, log))
            messages.extend(
                ToolMessage(content=replies[call["id"]], tool_call_id=call["id"])
                for call in response.tool_calls
            )
            if finished or replan is not None:
                break
        # The turn limit never skips an approved task: research what was never delegated.
        if replan is None:
            missing = self._unresearched(pending, rounds)
            if missing:
                for todo in missing:
                    rounds[todo.todo_id] += 1
                self._research_batch(
                    run_id,
                    [ResearchJob(todo, todo.research_question) for todo in missing],
                    thread_id,
                    claims,
                    results,
                    log,
                    automatic=True,
                )
        return results, replan

    def _decide(self, calls, by_id, rounds, pending, allow_replan, log: DecisionLog):
        """Validate one response's tool calls. Returns the replies for calls answered directly,
        the research jobs to run, whether research is complete, and any replan proposal."""
        replies: dict[str, str] = {}
        jobs: list[ResearchJob] = []
        finished = False
        replan: ReplanProposal | None = None
        batch: set[str] = set()
        for call in calls:
            name, args, call_id = call["name"], call.get("args") or {}, call["id"]
            if name == THINK_TOOL_NAME:
                reflection = str(args.get("reflection", ""))
                replies[call_id] = think(reflection)
                log.add("think", text=reflection)
            elif name == CONDUCT_RESEARCH_TOOL_NAME:
                todo_id = str(args.get("todo_id", "")).strip()
                topic = str(args.get("research_topic", "")).strip()
                if todo_id not in by_id:
                    replies[call_id] = (
                        f"Refused: unknown todo_id {todo_id!r}. Use one of: {', '.join(by_id)}."
                    )
                elif todo_id in batch:
                    replies[call_id] = f"Refused: task {todo_id} is already in this batch."
                elif rounds[todo_id] >= self.max_attempts_per_task:
                    replies[call_id] = (
                        f"Refused: task {todo_id} already had {rounds[todo_id]} research rounds."
                    )
                else:
                    batch.add(todo_id)
                    follow_up = rounds[todo_id] > 0
                    rounds[todo_id] += 1
                    jobs.append(
                        ResearchJob(by_id[todo_id], topic, follow_up=follow_up, call_id=call_id)
                    )
            elif name == RESEARCH_COMPLETE_TOOL_NAME:
                unresearched = [
                    todo
                    for todo in self._unresearched(pending, rounds)
                    if todo.todo_id not in batch
                ]
                if unresearched:
                    replies[call_id] = self._incomplete(unresearched)
                else:
                    replies[call_id] = "Research complete."
                    finished = True
                    log.add("complete")
            elif name == PROPOSE_REPLAN_TOOL_NAME and allow_replan:
                todo_ids = [str(tid) for tid in args.get("todo_ids") or [] if str(tid) in by_id]
                reason = str(args.get("reason", "")).strip()
                if not todo_ids or not reason:
                    replies[call_id] = "Refused: a replan needs a reason and known todo_ids."
                else:
                    replies[call_id] = "Replan proposed; a human will review a replacement plan."
                    replan = ReplanProposal(reason=reason, todo_ids=todo_ids)
                    log.add("replan", text=reason, todo_ids=todo_ids)
            else:
                replies[call_id] = f"Unknown tool {name!r}."
            if replies.get(call_id, "").startswith(("Refused", "Not complete", "Unknown tool")):
                log.add("refused", text=replies[call_id])
        return replies, jobs, finished, replan

    @staticmethod
    def _unresearched(pending, rounds) -> list[ResearchTodo]:
        return [todo for todo in pending if rounds[todo.todo_id] == 0]

    @staticmethod
    def _incomplete(unresearched: list[ResearchTodo]) -> str:
        return "Not complete: every task must be researched at least once. Still unresearched: " + (
            ", ".join(todo.todo_id for todo in unresearched)
        )

    @staticmethod
    def _task_list(pending: list[ResearchTodo], brief: str) -> str:
        lines = [f"Research brief: {brief}" if brief else "", "Pending research tasks:"]
        for todo in pending:
            lines.append(f"- todo_id: {todo.todo_id}\n  question: {todo.research_question}")
            if todo.evidence_needed:
                lines.append(f"  evidence needed: {todo.evidence_needed}")
            if todo.research_instruction:
                lines.append(f"  missing evidence to find: {todo.research_instruction}")
        return "\n".join(line for line in lines if line)

    # ----- research execution -----

    def _research_batch(
        self, run_id, jobs, thread_id, claims, results, log: DecisionLog, *, automatic=False
    ) -> dict[str, str]:
        """Run one batch of research jobs in parallel and persist them; returns the supervisor's
        reply per ConductResearch call id. `automatic` marks tasks the code delegated after the
        supervisor's turn limit."""
        previous = {job.todo.todo_id: results.get(job.todo.todo_id) for job in jobs}
        for job in jobs:
            if job.follow_up:
                job.todo.research_attempt += 1
                job.todo.research_instruction = job.topic
                job.todo.status = TodoStatus.PLANNED
        log.add(
            "delegate",
            automatic=automatic,
            tasks=[
                {
                    "todo_id": job.todo.todo_id,
                    "question": job.todo.research_question,
                    "round": job.todo.research_attempt,
                    "follow_up": job.follow_up,
                    "topic": job.topic,
                }
                for job in jobs
            ],
        )
        outcomes = self._run_sub_agents(run_id, jobs, thread_id, claims)
        replies = {}
        for job in jobs:
            todo_id = job.todo.todo_id
            todo, refs, chunks = outcomes[todo_id]
            summary = self._result_summary(run_id, todo)
            log.add("result", **{key: value for key, value in summary.items() if key != "findings"})
            if job.call_id:
                replies[job.call_id] = self._format_result(summary)
            earlier = previous[todo_id]
            if job.follow_up and earlier is not None:
                if todo.status == TodoStatus.FAILED:
                    # A failed follow-up keeps the earlier, usable evidence as the task's result.
                    todo.research_attempt -= 1
                    todo.status = TodoStatus.COMPLETED
                refs, chunks = [*earlier[1], *refs], [*earlier[2], *chunks]
            results[todo_id] = (todo, refs, chunks)
        return replies

    def _result_summary(self, run_id: str, todo: ResearchTodo) -> dict:
        """One task's outcome after a research round, read from what was persisted."""
        summary = {
            "todo_id": todo.todo_id,
            "question": todo.research_question,
            "round": todo.research_attempt,
            "status": todo.status.value,
        }
        try:
            bundle = json.loads(self.vfs.read(run_id, _evidence_path(todo)))
        except FileNotFoundError:
            try:
                failure = json.loads(self.vfs.read(run_id, f"{research_prefix(todo)}/outcome.json"))
                return {**summary, "error": str(failure.get("error", "unknown error"))}
            except FileNotFoundError:
                return {**summary, "error": ""}
        return {
            **summary,
            "sufficient": bool(bundle.get("sufficient")),
            "sources": len(bundle.get("sources", [])),
            "gaps": list(bundle.get("unresolved_gaps") or []),
            "findings": str(bundle.get("summary", "")),
        }

    @staticmethod
    def _format_result(summary: dict) -> str:
        """What the supervisor model reads about one task after a research round."""
        header = f"Task {summary['todo_id']} (round {summary['round']}): {summary['status']}"
        if "error" in summary:
            error = summary["error"]
            return (
                f"{header}\nNo usable evidence: {error}"
                if error
                else f"{header}\nNo usable evidence."
            )
        gaps = summary["gaps"]
        return (
            f"{header}\nSufficient: {'yes' if summary['sufficient'] else 'no'}\n"
            f"Sources: {summary['sources']}\n"
            f"Unresolved gaps: {'; '.join(gaps) if gaps else 'none'}\n"
            f"Findings:\n{summary['findings']}"
        )

    def _run_sub_agents(self, run_id: str, jobs: list[ResearchJob], thread_id: str, claims):
        """Synchronous entry point (called from the sync ``researching`` graph node): drives the
        async fan-out below to completion on a short-lived event loop, since nothing else in the
        call chain is async."""
        return asyncio.run(self._gather_sub_agents(run_id, jobs, thread_id, claims))

    async def _gather_sub_agents(
        self, run_id: str, jobs: list[ResearchJob], thread_id: str, claims: UrlClaims
    ) -> dict:
        # Each sub-agent's actual work (LLM calls, MCP search) is still synchronous, so
        # asyncio.to_thread runs it in a worker thread -- same real parallelism as before, capped
        # by the semaphore since asyncio.to_thread's own thread pool has no such limit. Unlike a
        # bare ThreadPoolExecutor, asyncio.to_thread propagates the calling contextvars.Context
        # into that thread automatically, so Langfuse/OpenTelemetry's "current span" and
        # LangChain's runnable config still reach each sub-agent's LLM call traces without any
        # manual context handling here.
        semaphore = asyncio.Semaphore(self.max_concurrent_researchers)

        async def bounded(job: ResearchJob) -> tuple[str, ResearchOutcome]:
            """Research one job while holding a concurrency slot."""
            async with semaphore:
                outcome = await asyncio.to_thread(
                    self._research_one, run_id, job, thread_id, claims
                )
                return job.todo.todo_id, outcome

        tasks = [asyncio.create_task(bounded(job)) for job in jobs]
        registry = UrlRegistry.load(self.vfs, run_id)
        by_id = {job.todo.todo_id: job.todo for job in jobs}
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
                    # Pages live in raw/<n>.md; the resume copy keeps only what the LLMs read.
                    payload["sources"] = [
                        {key: value for key, value in source.items() if key != "raw_content"}
                        for source in payload["sources"]
                    ]
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
        self, run_id: str, job: ResearchJob, thread_id: str, claims: UrlClaims
    ) -> ResearchOutcome:
        # One failing sub-agent must not cancel its siblings; the failure becomes its outcome.
        todo = job.todo
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
            outcome = self.researcher.research(
                self._sub_agent_question(job),
                run_id=run_id,
                thread_id=thread_id,
                task_id=todo.todo_id,
                claims=claims,
            )
            if job.follow_up and outcome.findings is not None:
                outcome.findings = self._with_earlier_findings(run_id, todo, outcome.findings)
            return outcome
        except Exception as exc:
            return failed_outcome(todo, f"Research sub-agent failed: {redact_error(exc)}")

    @staticmethod
    def _sub_agent_question(job: ResearchJob) -> str:
        """The sub-agent's only input: the task question plus the supervisor's focus."""
        todo = job.todo
        question = todo.research_question
        if job.follow_up:
            return f"{question}\n\nResearch only this missing evidence:\n{job.topic}"
        if todo.research_instruction:
            question += "\nResearch only this missing evidence: " + todo.research_instruction
        if job.topic and job.topic != todo.research_question:
            question += f"\n\nResearch focus from the supervisor:\n{job.topic}"
        return question

    def _with_earlier_findings(
        self, run_id: str, todo: ResearchTodo, findings: ResearchFindings
    ) -> ResearchFindings:
        """A follow-up only covers the gap: prepend the earlier round's findings."""
        earlier_todo = todo.model_copy(update={"research_attempt": todo.research_attempt - 1})
        try:
            earlier = json.loads(self.vfs.read(run_id, _evidence_path(earlier_todo)))
        except FileNotFoundError:
            return findings
        return findings.model_copy(
            update={"summary": f"{earlier.get('summary', '')}\n\n{findings.summary}".strip()}
        )
