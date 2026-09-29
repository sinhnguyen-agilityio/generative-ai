import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver

from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.graph.build import build_graph
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.storage.vfs import VFS
from tests.integration.test_langchain_scope import (
    RecordingChatModel,
    _brief_payload,
    _clear_payload,
    _guardrail_payload,
    _plan_payload,
)


def test_scope_clarification_checkpoint_resumes_after_process_restart(tmp_path):
    model = RecordingChatModel([
        _guardrail_payload(),
        {
            "need_clarification": True,
            "question": "Which versions and workload should be compared?",
            "verification": "",
        },
        _guardrail_payload(),
        _clear_payload(),
        _brief_payload(),
        _plan_payload(),
    ])
    checkpoint_path = tmp_path / "scope.sqlite"
    first_connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    graph = build_graph(
        checkpointer=SqliteSaver(first_connection),
        scope_agent=ScopeAgent(ModelRouter(model)),
        vfs=VFS(tmp_path / "vfs"),
    )
    config = {"configurable": {"thread_id": "scope-thread"}}
    first = graph.invoke({"run_id": "run", "thread_id": "scope-thread", "request": "Compare databases"}, config)
    assert first["stage"] == "Scoping"
    first_connection.close()

    second_connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
    restarted_graph = build_graph(
        checkpointer=SqliteSaver(second_connection),
        scope_agent=ScopeAgent(ModelRouter(model)),
        vfs=VFS(tmp_path / "vfs"),
    )
    resumed = restarted_graph.invoke(
        {"request": "Compare PostgreSQL and SQLite for local apps", "run_id": "run", "thread_id": "scope-thread"},
        config,
    )
    assert resumed["stage"] == "Planning"
    assert len(resumed["messages"]) == 4
    assert resumed["messages"][1].content.startswith("Which versions")
    assert resumed["messages"][-1].content == _clear_payload()["verification"]
    second_connection.close()
