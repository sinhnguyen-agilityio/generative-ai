"""Composition root: builds the live research workflow from settings and environment.

This is where concrete adapters are chosen (chat models, Tavily MCP, Chroma, SQLite checkpoints,
RAGAS judges) and injected into the agents and graph nodes. It has no Streamlit dependency:
``app.py`` caches these objects in the session and only renders the UI, and tests can build
the same graph with fakes through ``graph.build.build_graph`` directly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from pathlib import Path
import sqlite3

from intelligence_collector.agents.extension_judge import ExtensionJudge
from intelligence_collector.agents.feedback_router import FeedbackRouter
from intelligence_collector.agents.passage_selector import PassageSelector
from intelligence_collector.agents.report_judge import ReportJudge
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.config import RoleModelSettings, Settings
from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.nodes import ResearchNodes
from intelligence_collector.models.factory import ModelFactory, ModelRouters
from intelligence_collector.observability.evaluation import (
    make_context_precision_evaluator,
    make_faithfulness_evaluator,
    make_response_relevancy_evaluator,
)
from intelligence_collector.observability.tracing import RunTracer
from intelligence_collector.storage.chroma_vector_index import ChromaVectorIndex, default_embeddings
from intelligence_collector.storage.embedding_cache import CachingEmbeddings
from intelligence_collector.storage.run_registry import RunRegistry
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.tools.mcp import TavilyMCP


def build_model_routers() -> ModelRouters | None:
    """Reasoning-tier and standard-tier models, each with its own fallback; None until the
    reasoning model is configured. Tracing is not wired here: model calls inherit the Langfuse
    callback of the traced graph execution they run in."""
    if not os.getenv("INTELLIGENCE_REASONING_MODEL", "").strip():
        return None
    return ModelFactory().create_routers(RoleModelSettings.from_env())


def build_scope_agent(routers: ModelRouters) -> ScopeAgent:
    # Scope runs on the reasoning tier; its content-safety guardrail is a cheap binary check.
    return ScopeAgent(routers.reasoning, guardrail_router=routers.standard)


def open_checkpoint_connection(checkpoint_path: str) -> sqlite3.Connection:
    path = Path(checkpoint_path).expanduser()
    if path.exists() and path.is_dir():
        raise RuntimeError(f"Checkpoint path points to a directory, not a SQLite file: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(path, check_same_thread=False)
    except (OSError, sqlite3.Error) as exc:
        raise RuntimeError(
            f"Cannot open SQLite checkpoint database at '{path}'. "
            "Check INTELLIGENCE_CHECKPOINT_PATH and directory write permissions."
        ) from exc


def _judge_key_env(settings: Settings) -> str:
    return (
        "GOOGLE_API_KEY" if settings.faithfulness_provider == "google_genai" else "OPENAI_API_KEY"
    )


def quality_evaluators(settings: Settings, embeddings) -> dict:
    """RAGAS evaluators keyed by their ``ResearchNodes`` argument. Faithfulness is the third
    verification gate, so it is always built when the judge key is set; Context Precision and
    Response Relevancy follow ``INTELLIGENCE_EVAL_METRICS``. The judge (default OpenAI
    gpt-4.1-mini) is independent of the app's reasoning/standard tier models. Without the
    judge key no evaluator exists and verification reports ERROR (see ``require_faithfulness``).
    Response Relevancy reuses the vector index's embeddings."""
    api_key = os.getenv(_judge_key_env(settings), "").strip()
    if not api_key:
        return {}
    judge = {
        "model": settings.faithfulness_model,
        "api_key": api_key,
        "provider": settings.faithfulness_provider,
    }
    factories = {
        "faithfulness": ("faithfulness_evaluator", lambda: make_faithfulness_evaluator(**judge)),
        "context_precision": (
            "context_precision_evaluator",
            lambda: make_context_precision_evaluator(**judge),
        ),
        "response_relevancy": (
            "response_relevancy_evaluator",
            lambda: make_response_relevancy_evaluator(**judge, embeddings=embeddings),
        ),
    }
    metrics = dict.fromkeys(["faithfulness", *settings.eval_metrics])
    return {factories[m][0]: factories[m][1]() for m in metrics}


def _research_nodes(
    settings: Settings,
    routers: ModelRouters,
    agent: ScopeAgent,
    vfs: VFS,
    warn: Callable[[str], None],
) -> ResearchNodes | None:
    """The post-planning nodes, or None (scope-and-plan only) without Tavily configured."""
    tavily_key = os.getenv("TAVILY_API_KEY", "").strip()
    tavily_url = os.getenv("TAVILY_MCP_URL", "").strip()
    if not (tavily_key and tavily_url):
        warn(
            "Research is disabled: configure TAVILY_API_KEY and TAVILY_MCP_URL. "
            "This run can only clarify the request and prepare a plan."
        )
        return None
    judge_key = _judge_key_env(settings)
    if not os.getenv(judge_key, "").strip():
        warn(
            f"Report verification needs RAGAS Faithfulness: configure {judge_key}. "
            "Until then every verification ends in ERROR and no report can be approved."
        )
    embeddings = default_embeddings(settings.embedding_model, settings.embedding_provider)
    # Passage selection and the vector index embed the same page chunks: the index reuses the
    # vectors selection computed instead of embedding them twice.
    chunk_embeddings = CachingEmbeddings(embeddings)
    researcher = ResearchAgent(
        routers.standard,
        # Each research sub-agent gets its own MCP client and per-task search budget.
        mcp_factory=lambda: TavilyMCP(tavily_url, tavily_key, max_calls=settings.max_search_calls),
        max_search_calls=settings.max_search_calls,
        max_sources=settings.max_sources_per_task,
        passage_selector=PassageSelector(chunk_embeddings),
    )
    return ResearchNodes(
        vfs,
        # Real semantic retrieval, persisted locally via Chroma. Used to narrow the Build/Fix
        # prompt to each question's most relevant sources; Verify always re-reads the full
        # evidence independently, so this never affects what gets checked.
        ChromaVectorIndex(
            settings.chroma_path,
            embeddings=chunk_embeddings,
            model=settings.embedding_model,
            provider=settings.embedding_provider,
        ),
        researcher,
        # Coverage gate; shares the scope/planner reasoning-tier model. Research readiness is
        # derived from the stored bundles without an extra assessor call.
        report_judge=ReportJudge(agent.router),
        report_guidelines=settings.report_guidelines,
        max_concurrent_researchers=settings.max_concurrent_researchers,
        recovery_budget=settings.recovery_attempts,
        evidence_top_k=settings.evidence_top_k,
        require_faithfulness=True,
        **quality_evaluators(settings, embeddings),
    )


@dataclass
class ResearchRuntime:
    """The live graph and the durable stores it owns for one app session."""

    graph: object
    vfs: VFS
    run_registry: RunRegistry
    checkpoint_connection: sqlite3.Connection


def build_runtime(
    routers: ModelRouters,
    agent: ScopeAgent,
    tracer: RunTracer,
    *,
    lifecycle: Callable[[str, str], None],
    warn: Callable[[str], None] = lambda _message: None,
) -> ResearchRuntime:
    """Wire the full research graph: scope, planning with approval, research when Tavily is
    configured, build-verify-fix and human review, checkpointed in SQLite."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = Settings.from_env()
    connection = open_checkpoint_connection(settings.checkpoint_path)
    run_registry = RunRegistry(settings.run_registry_path)
    vfs = VFS(settings.vfs_root)
    graph = build_graph(
        scope_agent=agent,
        vfs=vfs,
        research_nodes=_research_nodes(settings, routers, agent, vfs, warn),
        checkpointer=SqliteSaver(connection),
        tracer=tracer,
        lifecycle=lifecycle,
        run_registry=run_registry,
        with_approval=True,
        output_dir=settings.output_dir,
        # A cheap binary classification, so the standard tier (fewer rate-limit failures).
        feedback_router=FeedbackRouter(routers.standard),
        # Decides whether stored evidence already covers a section the reviewer asked to add.
        extension_judge=ExtensionJudge(routers.standard),
    )
    return ResearchRuntime(graph, vfs, run_registry, connection)
