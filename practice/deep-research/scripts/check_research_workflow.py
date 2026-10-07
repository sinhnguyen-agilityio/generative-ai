"""Opt-in live workflow test; stops at the first human interrupt, never approves."""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--no-output-caps", action="store_true")
    parser.add_argument("--resume-run", help="Continue an unfinished diagnostic checkpoint")
    parser.add_argument("--question", default="What is the latest stable version of Python?")
    args = parser.parse_args()
    if not args.run_live:
        parser.error("Pass --run-live to authorize paid model, search and embedding calls")
    sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    if args.no_output_caps:
        os.environ["INTELLIGENCE_DISABLE_OUTPUT_CAPS"] = "1"

    from langchain_core.callbacks import BaseCallbackHandler
    from langgraph.checkpoint.sqlite import SqliteSaver
    from intelligence_collector.runtime import quality_evaluators
    from intelligence_collector.agents.scope import ScopeAgent
    from intelligence_collector.agents.researcher import ResearchAgent
    from intelligence_collector.agents.report_judge import ReportJudge
    from intelligence_collector.agents.claim_judge import ClaimJudge
    from intelligence_collector.config import Settings, RoleModelSettings
    from intelligence_collector.graph.build import build_graph
    from intelligence_collector.graph.nodes import ResearchNodes
    from intelligence_collector.models.factory import ModelFactory
    from intelligence_collector.models.redaction import redact_text
    from intelligence_collector.storage.vfs import VFS
    from intelligence_collector.storage.chroma_vector_index import (
        ChromaVectorIndex,
        default_embeddings,
    )
    from intelligence_collector.tools.mcp import TavilyMCP

    calls = []

    class Usage(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs):
            for generations in response.generations:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    usage = getattr(message, "usage_metadata", None)
                    if usage:
                        calls.append(dict(usage))
                        print("USAGE", json.dumps(usage), flush=True)

    settings = Settings.from_env()
    roles = RoleModelSettings.from_env()
    routers = ModelFactory().create_routers(roles, callbacks=[Usage()])
    run_id = args.resume_run or str(uuid4())
    if str(UUID(run_id)) != run_id:
        parser.error("resume-run must be a UUID")
    # Keep diagnostic paths short enough for Windows' legacy path-length limit.
    root = Path(".tmp/lt") / run_id[:8]
    legacy_root = Path(".data/live-tests") / run_id
    if args.resume_run and legacy_root.exists() and not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        legacy_root.rename(root)
    root.mkdir(parents=True, exist_ok=True)
    vfs = VFS(root / "vfs")
    embeddings = default_embeddings(settings.embedding_model, settings.embedding_provider)
    researcher = ResearchAgent(
        routers.standard,
        mcp_factory=lambda: TavilyMCP(
            os.environ["TAVILY_MCP_URL"],
            os.environ["TAVILY_API_KEY"],
            max_calls=settings.max_search_calls,
        ),
        max_search_calls=settings.max_search_calls,
        max_sources=settings.max_sources_per_task,
    )
    nodes = ResearchNodes(
        vfs,
        ChromaVectorIndex(
            str(root / "chroma"),
            embeddings=embeddings,
            model=settings.embedding_model,
            provider=settings.embedding_provider,
        ),
        researcher,
        report_judge=ReportJudge(routers.reasoning),
        require_faithfulness=True,
        claim_judge=ClaimJudge(routers.reasoning),
        report_guidelines=settings.report_guidelines,
        max_concurrent_researchers=settings.max_concurrent_researchers,
        recovery_budget=settings.recovery_attempts,
        evidence_top_k=settings.evidence_top_k,
        **quality_evaluators(settings, embeddings),
    )
    started = time.monotonic()
    with sqlite3.connect(root / "checkpoints.sqlite", check_same_thread=False) as connection:
        graph = build_graph(
            scope_agent=ScopeAgent(routers.reasoning, guardrail_router=routers.standard),
            research_nodes=nodes,
            vfs=vfs,
            checkpointer=SqliteSaver(connection),
            with_approval=True,
            output_dir=str(root / "exports"),
            lifecycle=lambda node, status: print(
                f"{time.monotonic() - started:.1f}s {node}: {status}", flush=True
            ),
        )
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 100}
        print(
            f"RUN {run_id}; no_output_caps={args.no_output_caps}; "
            f"reasoning_model={roles.reasoning_model}; standard_model={roles.standard_model}",
            flush=True,
        )
        error = None
        try:
            graph.invoke(
                None
                if args.resume_run
                else {"run_id": run_id, "thread_id": run_id, "request": args.question},
                config,
            )
        except Exception as exc:
            chain = []
            while exc is not None:
                chain.append(f"{type(exc).__name__}: {redact_text(exc)}")
                exc = exc.__cause__
            error = " <- ".join(chain)
        snapshot = graph.get_state(config)
        state = snapshot.values
        summary = {
            key: state.get(key)
            for key in (
                "stage",
                "research_status",
                "result_incomplete",
                "verification",
                "verifier_error",
                "last_error",
                "research_limitations",
                "draft",
                "scoped_request",
                "todos",
            )
        }
        summary.update(
            error=error,
            next_nodes=snapshot.next,
            calls=calls,
            elapsed_seconds=time.monotonic() - started,
        )
        vfs.write_json(run_id, "diagnostic.json", summary, "diagnostic")
        print(json.dumps(summary, default=str, ensure_ascii=False, indent=2), flush=True)
        print(f"Artifacts: {root.resolve()}", flush=True)
        return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
