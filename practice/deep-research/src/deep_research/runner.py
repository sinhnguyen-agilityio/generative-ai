"""Runs the graph for one thread: start or resume a turn, and read where the thread stands."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from deep_research import tracing
from deep_research.config import Settings
from deep_research.agents.guardrail import check_input
from deep_research.orchestrator import build_graph
from deep_research.models import Tier
from deep_research.search import PageStore, TavilySearch


@dataclass
class ThreadView:
    values: dict
    pending: dict | None  # the interrupt payload waiting for a human decision
    unfinished: bool  # stopped mid-run (error or closed tab); can be continued


class DeepResearch:
    def __init__(self, settings: Settings, graph=None):
        self.settings = settings
        self.loop = asyncio.Runner()
        if graph is None:
            Path(settings.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

            async def connect():
                return AsyncSqliteSaver(await aiosqlite.connect(settings.checkpoint_path))

            checkpointer = self.loop.run(connect())
            graph = build_graph(
                settings,
                reasoning=Tier(settings.reasoning),
                standard=Tier(settings.standard),
                search=TavilySearch(settings.tavily_url, settings.tavily_api_key, settings.search_max_results),
                store=PageStore(settings.chroma_path, settings.embedding_model, settings.embedding_provider),
                checkpointer=checkpointer,
            )
        self.graph = graph

    def _config(self, thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100,
            **tracing.langfuse_config(self.settings.langfuse_enabled, thread_id),
        }

    def _stream(self, thread_id: str, payload) -> Iterator[tuple[str, dict]]:
        """Yields ``(node, state update)`` as each node finishes."""
        try:
            stream = self.graph.astream(
                payload, self._config(thread_id), stream_mode="updates", subgraphs=True
            )
            while (item := self.loop.run(anext(stream, None))) is not None:
                namespace, chunk = item
                # Expose supervisor rounds for progress, but keep researcher internals private.
                if namespace and (len(namespace) != 1 or not namespace[0].startswith("supervisor:")):
                    continue
                for node, update in chunk.items():
                    if node != "__interrupt__" and (namespace or node != "supervisor"):
                        yield node, update or {}
        finally:
            self.loop.run(stream.aclose())
            tracing.flush(self.settings.langfuse_enabled)

    def ask(self, thread_id: str, text: str):
        """Raises ``GuardrailViolation`` before anything is stored when the text is rejected."""
        return self._stream(thread_id, {"messages": [HumanMessage(check_input(text))]})

    def resume(self, thread_id: str, decision: dict):
        """Raises ``GuardrailViolation`` when the reviewer's feedback, topics or criteria are rejected."""
        decision = dict(decision)
        if decision.get("feedback"):
            decision["feedback"] = check_input(decision["feedback"])
        for key in ("topics", "criteria"):
            if decision.get(key):
                decision[key] = [check_input(t) for t in decision[key] if t.strip()]
        return self._stream(thread_id, Command(resume=decision))

    def continue_run(self, thread_id: str):
        return self._stream(thread_id, None)

    def view(self, thread_id: str) -> ThreadView:
        snapshot = self.loop.run(self.graph.aget_state({"configurable": {"thread_id": thread_id}}))
        pending = snapshot.interrupts[0].value if snapshot.interrupts else None
        return ThreadView(snapshot.values, pending, unfinished=bool(snapshot.next) and pending is None)
