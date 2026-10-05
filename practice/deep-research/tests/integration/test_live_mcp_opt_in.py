import json
import os

import pytest
from dotenv import load_dotenv

from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.worker import run_worker
from intelligence_collector.contracts import ResearchTodo, TodoStatus
from intelligence_collector.models.factory import ModelConfig, ModelFactory
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import TavilyMCP


def _live_tavily_configuration():
    load_dotenv()
    if os.getenv("RUN_LIVE_MCP", "0").strip() != "1":
        pytest.skip("set RUN_LIVE_MCP=1 to run the live Tavily MCP acceptance test")
    endpoint = os.getenv("TAVILY_MCP_URL", "").strip()
    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    model_name = os.getenv("INTELLIGENCE_REASONING_MODEL", "").strip()
    if not endpoint or not api_key or not model_name:
        pytest.skip(
            "TAVILY_MCP_URL, TAVILY_API_KEY, and INTELLIGENCE_REASONING_MODEL are required "
            "for live MCP"
        )
    return endpoint, api_key, model_name


def test_live_tavily_mcp_preserves_provenance_artifacts_and_index(stores):
    endpoint, api_key, model_name = _live_tavily_configuration()
    vfs, index = stores
    mcp = TavilyMCP(endpoint, api_key, max_calls=5)
    model = ModelFactory().create(
        ModelConfig(
            model_name,
            provider=os.getenv("INTELLIGENCE_REASONING_PROVIDER") or None,
            timeout=45,
        )
    )
    researcher = ResearchAgent(ModelRouter(model), lambda: mcp)
    todo = ResearchTodo(research_question="What is the official Python 3.13 release date?")

    completed, refs, chunks = run_worker(vfs, index, "live-tavily", todo, researcher)

    assert completed.status == TodoStatus.COMPLETED
    assert 1 <= mcp.calls <= 5
    assert len(mcp.requests) == mcp.calls
    assert all(request["server"] == "tavily" for request in mcp.requests)
    assert all(request["tool"] == "tavily_search" for request in mcp.requests)
    assert {ref.kind for ref in refs} >= {
        "raw_capture",
        "notes",
        "excerpts",
        "evidence_bundle",
    }
    assert chunks
    assert all(chunk.metadata["run_id"] == "live-tavily" for chunk in chunks)
    hits = index.retrieve(todo.research_question, limit=10, run_id="live-tavily")
    assert hits and {hit.chunk_id for hit in hits} <= {chunk.chunk_id for chunk in chunks}
    # Tavily's full pages are stored in the VFS; the LLMs only read their short summaries.
    pages = [ref for ref in refs if ref.kind == "raw_page"]
    assert pages
    for source in json.loads(
        vfs.read("live-tavily", f"research/{todo.todo_id}/evidence/result.json")
    )["sources"]:
        if source.get("raw_path"):
            page = vfs.read("live-tavily", source["raw_path"]).decode()
            assert len(source["excerpt"]) < len(page)

    evidence = json.loads(
        vfs.read(
            "live-tavily",
            f"research/{todo.todo_id}/evidence/result.json",
        )
    )
    assert evidence["search_calls"] <= 5
    assert evidence["sources"]
    assert all(source["mcp_server"] == "tavily" for source in evidence["sources"])
    assert all(source["mcp_tool"] == "tavily_search" for source in evidence["sources"])
    serialized = json.dumps({"requests": mcp.requests, "evidence": evidence})
    assert api_key not in serialized
    assert api_key not in repr(mcp)


def test_live_supervisor_delegates_every_task_with_its_tools(stores):
    endpoint, api_key, model_name = _live_tavily_configuration()
    vfs, index = stores
    from intelligence_collector.agents.supervisor import ResearchSupervisor

    model = ModelFactory().create(
        ModelConfig(
            model_name,
            provider=os.getenv("INTELLIGENCE_REASONING_PROVIDER") or None,
            timeout=45,
        )
    )
    decisions = []
    router = ModelRouter(
        model,
        trace=lambda **event: decisions.append(event) if event["role"] == "supervisor" else None,
    )
    researcher = ResearchAgent(router, lambda: TavilyMCP(endpoint, api_key, max_calls=2))
    todos = [
        ResearchTodo(research_question="What is the official Python 3.13 release date?"),
        ResearchTodo(research_question="What is the latest stable LangGraph Python version?"),
    ]

    results = ResearchSupervisor(researcher, vfs, index).delegate(
        "live-supervisor", todos, brief="Python 3.13 release date and the latest LangGraph version"
    )

    assert [todo.status for todo, _, _ in results] == [TodoStatus.COMPLETED] * 2
    assert decisions  # the supervisor model made the decisions
    called = {call["name"] for event in decisions for call in event["decision"]["tool_calls"]}
    assert "ConductResearch" in called
