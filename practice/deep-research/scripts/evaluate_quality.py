"""Opt-in live quality evaluation: run each case in evals/cases.json through the real workflow.

Each case starts from the user question, answers a scope clarification and approves the
initial plan automatically (both recorded), and stops at the first report-approval or
recovery pause. It never approves or publishes a report. Saved per case under
evals/runs/<label>/<case_id>/: report.md, summary.json (stage, verification, model calls and
tokens per role, Tavily searches, vector retrievals, node latency) and evidence.json (the
sources collected in that run, for judging the report against its rubric).

Paid model, search and embedding calls are made only with --run-live.
"""

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _interrupt_kind(snapshot) -> str:
    for task in snapshot.tasks or ():
        for item in task.interrupts or ():
            if isinstance(item.value, dict):
                return str(item.value.get("kind") or "")
    return ""


def run_case(case: dict, label: str) -> dict:
    from langchain_core.callbacks import BaseCallbackHandler
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command

    from intelligence_collector.agents.feedback_router import FeedbackRouter
    from intelligence_collector.agents.report_judge import ReportJudge
    from intelligence_collector.agents.researcher import ResearchAgent
    from intelligence_collector.agents.scope import ScopeAgent
    from intelligence_collector.config import RoleModelSettings, Settings
    from intelligence_collector.graph.build import build_graph
    from intelligence_collector.graph.nodes import ResearchNodes
    from intelligence_collector.models.factory import ModelFactory
    from intelligence_collector.models.redaction import redact_text
    from intelligence_collector.observability.evaluation import make_faithfulness_evaluator
    from intelligence_collector.source_verification import load_catalog
    from intelligence_collector.storage.chroma_vector_index import ChromaVectorIndex, default_embeddings
    from intelligence_collector.storage.vfs import VFS
    from intelligence_collector.tools.mcp import TavilyMCP

    calls: Counter = Counter()
    tokens: dict = defaultdict(lambda: {"input": 0, "output": 0})
    roles_by_run: dict = {}

    class Usage(BaseCallbackHandler):
        def on_chat_model_start(self, serialized, messages, *, run_id, metadata=None, **kwargs):
            roles_by_run[run_id] = (metadata or {}).get("role", "unknown")

        def on_llm_end(self, response, *, run_id, **kwargs):
            role = roles_by_run.get(run_id, "unknown")
            for generations in response.generations:
                for generation in generations:
                    usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                    tokens[role]["input"] += int(usage.get("input_tokens") or 0)
                    tokens[role]["output"] += int(usage.get("output_tokens") or 0)

    def trace(**event):
        calls[event["role"]] += 1

    settings = Settings.from_env()
    roles = RoleModelSettings.from_env()
    routers = ModelFactory().create_routers(roles, trace=trace, callbacks=[Usage()])
    run_id = str(uuid4())
    # Keep run storage short: nested VFS paths under the repository exceed Windows' legacy
    # 260-character path limit.
    root = Path(tempfile.gettempdir()) / "dr-eval" / label / case["id"]
    root.mkdir(parents=True, exist_ok=True)
    vfs = VFS(root / run_id[:8] / "vfs")
    embeddings = default_embeddings(settings.embedding_model, settings.embedding_provider)
    index = ChromaVectorIndex(str(root / run_id[:8] / "chroma"), embeddings=embeddings,
                              model=settings.embedding_model, provider=settings.embedding_provider)
    counters = Counter()
    retrieve = index.retrieve

    def counted_retrieve(*args, **kwargs):
        counters["vector_retrievals"] += 1
        return retrieve(*args, **kwargs)

    index.retrieve = counted_retrieve

    class CountedTavily(TavilyMCP):
        def search(self, *args, **kwargs):
            counters["tavily_searches"] += 1
            return super().search(*args, **kwargs)

    researcher = ResearchAgent(
        routers.standard,
        mcp_factory=lambda: CountedTavily(os.environ["TAVILY_MCP_URL"], os.environ["TAVILY_API_KEY"],
                                          max_calls=settings.max_search_calls),
        max_search_calls=settings.max_search_calls, max_sources=settings.max_sources_per_task)
    evaluators = {}
    judge_key = "GOOGLE_API_KEY" if settings.faithfulness_provider == "google_genai" else "OPENAI_API_KEY"
    if os.getenv(judge_key, "").strip():
        faithfulness = make_faithfulness_evaluator(
            model=settings.faithfulness_model, api_key=os.environ[judge_key],
            provider=settings.faithfulness_provider)

        def counted_faithfulness(*args, **kwargs):
            counters["ragas_faithfulness_runs"] += 1
            return faithfulness(*args, **kwargs)

        evaluators["faithfulness_evaluator"] = counted_faithfulness
    nodes = ResearchNodes(
        vfs, index, researcher,
        # Mirrors the app wiring (app.py::_get_scope_graph). The "before" baseline was run on the
        # previous wiring, which also passed research_assessor=ResearchEvidenceJudge(...).
        report_judge=ReportJudge(routers.reasoning),
        report_guidelines=settings.report_guidelines,
        require_faithfulness=True,
        max_concurrent_researchers=settings.max_concurrent_researchers,
        recovery_budget=settings.recovery_attempts, evidence_top_k=settings.evidence_top_k,
        **evaluators)
    node_seconds: dict = defaultdict(float)
    node_runs: Counter = Counter()
    started_at: dict = {}

    def lifecycle(node, status):
        if status == "running":
            started_at[node] = time.monotonic()
            node_runs[node] += 1
        elif node in started_at:
            node_seconds[node] += time.monotonic() - started_at.pop(node)

    started = time.monotonic()
    pauses, error = [], None
    with sqlite3.connect(root / run_id[:8] / "checkpoints.sqlite", check_same_thread=False) as connection:
        graph = build_graph(
            scope_agent=ScopeAgent(routers.reasoning, guardrail_router=routers.standard),
            research_nodes=nodes, vfs=vfs, checkpointer=SqliteSaver(connection),
            with_approval=True, output_dir=str(root / "exports"), lifecycle=lifecycle,
            feedback_router=FeedbackRouter(routers.standard))
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 100}
        graph_input = {"run_id": run_id, "thread_id": run_id, "request": case["question"]}
        try:
            for _ in range(6):
                graph.invoke(graph_input, config)
                snapshot = graph.get_state(config)
                kind = _interrupt_kind(snapshot)
                pauses.append(kind or ",".join(snapshot.next) or "end")
                if kind == "clarification_required":
                    graph_input = Command(resume={"reply": case["clarification_reply"], "actor": "eval"})
                elif kind == "plan_approval":
                    graph_input = Command(resume={"approved": True, "actor": "eval"})
                else:
                    break
        except Exception as exc:
            chain = []
            while exc is not None:
                chain.append(f"{type(exc).__name__}: {redact_text(exc)}")
                exc = exc.__cause__
            error = " <- ".join(chain)
        state = graph.get_state(config).values
    elapsed = time.monotonic() - started
    out = ROOT / "evals" / "runs" / label / case["id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(state.get("draft") or "(no report)", encoding="utf-8")
    try:
        catalog = load_catalog(vfs, run_id, state.get("evidence", []))
        evidence = {
            "sources": {sid: {"url": s.url, "title": s.title, "excerpt": s.excerpt[:1500]}
                        for sid, s in catalog.sources.items()},
            "bundles": [{"question": b.question, "source_ids": b.source_ids, "summary": b.summary,
                         "gaps": [g.text for g in b.gaps]} for b in catalog.bundles],
        }
    except Exception as exc:  # evidence is diagnostic output only
        evidence = {"error": redact_text(exc)}
    (out / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    verification = state.get("verification") or {}
    summary = {
        "case_id": case["id"], "label": label, "run_id": run_id, "question": case["question"],
        "stage": state.get("stage"), "pauses": pauses, "error": error,
        "brief": state.get("scoped_request"), "todos": [
            {key: todo.get(key) for key in ("research_question", "evidence_needed", "status", "research_attempt")}
            for todo in state.get("todos", [])],
        "plan": state.get("plan"),
        "research_status": state.get("research_status"),
        "research_limitations": state.get("research_limitations"),
        "verification_passed": verification.get("passed"),
        "verification_outcome": state.get("verification_outcome"),
        "failed_checks": [c for c in verification.get("check_results", []) if not c.get("passed")],
        "verifier_error": state.get("verifier_error"), "last_error": state.get("last_error"),
        "faithfulness_score": state.get("faithfulness_score"),
        "draft_version": state.get("draft_version"),
        "improvement_cycles": state.get("recovery_attempt", 0),
        "repair_history": state.get("bvf_history", []),
        "model_calls_by_role": dict(calls), "tokens_by_role": dict(tokens),
        "tavily_searches": counters["tavily_searches"],
        "vector_retrievals": counters["vector_retrievals"],
        "ragas_faithfulness_runs": counters["ragas_faithfulness_runs"],
        "node_runs": dict(node_runs),
        "node_seconds": {k: round(v, 1) for k, v in node_seconds.items()},
        "elapsed_seconds": round(elapsed, 1),
        "models": {"reasoning": roles.reasoning_model, "standard": roles.standard_model,
                   "faithfulness": settings.faithfulness_model},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                                      encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--label", required=True, help="Output folder, e.g. before or after")
    parser.add_argument("--case", action="append", help="Case id to run (default: all)")
    args = parser.parse_args()
    if not args.run_live:
        parser.error("Pass --run-live to authorize paid model, search and embedding calls")
    sys.stdout.reconfigure(encoding="utf-8")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    cases = json.loads((ROOT / "evals" / "cases.json").read_text(encoding="utf-8"))["cases"]
    for case in cases:
        if args.case and case["id"] not in args.case:
            continue
        summary = run_case(case, args.label)
        print(json.dumps({key: summary[key] for key in (
            "case_id", "stage", "pauses", "error", "verification_passed", "draft_version",
            "improvement_cycles", "model_calls_by_role", "tavily_searches", "vector_retrievals",
            "elapsed_seconds")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
