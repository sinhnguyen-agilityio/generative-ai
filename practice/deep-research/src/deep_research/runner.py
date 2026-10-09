"""Runs the graph for one thread: start or resume a turn, and read where the thread stands."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import aiosqlite
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from deep_research.orchestrator import build_graph
from deep_research.page_store import PageStore
from deep_research.settings import Settings
from deep_research.tavily_search import TavilySearch
from deep_research.thread_view import ThreadView
from deep_research.tier import Tier
from deep_research.utils import tracing
from deep_research.utils.guardrail import check_input


class DeepResearch:
    def __init__(self, settings: Settings, graph=None):
        """Initialize a synchronous runner backed by a persistent async event loop.

        Args:
            settings: Runtime settings for services, checkpoints, and tracing.
            graph: Optional compiled graph; omitted to build the production workflow.
        """
        self.settings = settings
        self.loop = asyncio.Runner()
        if graph is None:
            Path(settings.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

            async def connect():
                """Open the SQLite checkpointer inside the runner event loop.

                Returns:
                    Async SQLite saver using the configured checkpoint path.
                """
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
        """Build execution configuration for a persistent research thread.

        Args:
            thread_id: Identifier shared by checkpoints and optional tracing.

        Returns:
            Thread configuration, recursion limit, and optional Langfuse callbacks.
        """
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 100,
            **tracing.langfuse_config(self.settings.langfuse_enabled, thread_id),
        }

    def _stream(self, thread_id: str, payload) -> Iterator[tuple[str, dict]]:
        """Stream completed graph nodes through the synchronous runner API.

        Expose supervisor rounds while hiding researcher internals. Close the async stream
        and flush enabled tracing when iteration finishes or fails.

        Args:
            thread_id: Research thread to start or resume.
            payload: Initial state, resume command, or None to continue interrupted work.

        Yields:
            Pairs of node name and state update; human interrupts are read through view.
        """
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
        """Validate a user message and prepare a new graph turn.

        Args:
            thread_id: Research thread receiving the message.
            text: User request or answer to a clarification question.

        Returns:
            Iterator of node updates; graph execution starts when it is consumed.

        Raises:
            GuardrailViolation: The message is empty, oversized, or contains a detected credential.
        """
        return self._stream(thread_id, {"messages": [HumanMessage(check_input(text))]})

    def resume(self, thread_id: str, decision: dict):
        """Validate a human review decision and prepare to resume the waiting graph.

        Args:
            thread_id: Research thread waiting at a review interrupt.
            decision: Approval, feedback, and optional edited topics or criteria.

        Returns:
            Iterator of node updates after the review decision.

        Raises:
            GuardrailViolation: Feedback, topics, or criteria fail input validation.
        """
        decision = dict(decision)
        if decision.get("feedback"):
            decision["feedback"] = check_input(decision["feedback"])
        for key in ("topics", "criteria"):
            if decision.get(key):
                decision[key] = [check_input(t) for t in decision[key] if t.strip()]
        return self._stream(thread_id, Command(resume=decision))

    def continue_run(self, thread_id: str):
        """Continue checkpointed work after a run stopped before completion.

        Args:
            thread_id: Research thread whose pending graph work should continue.

        Returns:
            Iterator of node updates from the saved execution point.
        """
        return self._stream(thread_id, None)

    def view(self, thread_id: str) -> ThreadView:
        """Read saved thread state and distinguish human review from unfinished work.

        Args:
            thread_id: Research thread to inspect.

        Returns:
            Thread values, first pending review payload, and unfinished-run flag.
        """
        snapshot = self.loop.run(self.graph.aget_state({"configurable": {"thread_id": thread_id}}))
        pending = snapshot.interrupts[0].value if snapshot.interrupts else None
        return ThreadView(snapshot.values, pending, unfinished=bool(snapshot.next) and pending is None)
