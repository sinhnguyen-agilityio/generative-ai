"""Typed boundaries between the workflow and its collaborators.

Nodes and agents depend on these Protocols rather than on concrete adapters, so tests can inject
fakes and a backend can be swapped without touching the workflow:

* ``ModelRouterPort`` structured and tool-calling model calls (``models.router.ModelRouter``)
* ``ArtifactStore``   run-scoped artifact storage (``storage.vfs.VFS``)
* ``VectorIndex``     passage index and retrieval (``storage.vector_index``, ``storage.chroma_*``)
* ``MCPResearchPort`` external search (``tools.mcp.ControlledMCP``, ``tools.mcp.TavilyMCP``)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from intelligence_collector.contracts import ArtifactRef
from intelligence_collector.storage.vector_index import VectorIndex
from intelligence_collector.tools.mcp import MCPResearchPort

T = TypeVar("T", bound=BaseModel)


class ModelRouterPort(Protocol):
    """One model tier: validated structured output, or a tool-calling decision."""

    model_name: str

    def invoke_messages(
        self,
        role: str,
        messages: list[BaseMessage],
        schema: type[T],
        *,
        run_id: str = "",
        thread_id: str = "",
    ) -> T: ...

    def invoke_with_tools(
        self,
        role: str,
        messages: list[BaseMessage],
        tools: Sequence[Any],
        *,
        run_id: str = "",
        thread_id: str = "",
        tool_choice: str | None = None,
        attempt: int = 1,
    ) -> AIMessage: ...


class ArtifactStore(Protocol):
    """Path-safe storage of one run's artifacts under ``<root>/<run_id>/``."""

    root: Path

    def write(
        self,
        run_id: str,
        relative: str,
        content: str | bytes,
        kind: str,
        *,
        task_id: str | None = None,
    ) -> ArtifactRef: ...

    def write_json(
        self,
        run_id: str,
        relative: str,
        value: Any,
        kind: str,
        *,
        task_id: str | None = None,
    ) -> ArtifactRef: ...

    def read(self, run_id: str, relative: str) -> bytes: ...


__all__ = ["ArtifactStore", "MCPResearchPort", "ModelRouterPort", "VectorIndex"]
