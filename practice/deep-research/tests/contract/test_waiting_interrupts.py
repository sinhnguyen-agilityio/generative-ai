from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from intelligence_collector.graph.nodes import await_clarification, await_recovery
from intelligence_collector.graph.state import GraphState


def _one_node_graph(name, node):
    builder = StateGraph(GraphState)
    builder.add_node(name, node)
    builder.add_edge(START, name)
    builder.add_edge(name, END)
    return builder.compile(checkpointer=MemorySaver())


def test_clarification_wait_is_a_resumable_interrupt():
    graph = _one_node_graph("await_clarification", await_clarification)
    config = {"configurable": {"thread_id": "scope-thread"}}
    paused = graph.invoke({
        "run_id": "run",
        "thread_id": "scope-thread",
        "stage": "Scoping",
        "scope_assessment": {
            "need_clarification": True,
            "question": "Which workload?",
            "verification": "",
        },
    }, config)

    assert paused.get("__interrupt__")
    assert paused["__interrupt__"][0].value["kind"] == "clarification_required"
    resumed = graph.invoke(
        Command(resume={"reply": "A local read-heavy workload", "actor": "tester"}),
        config,
    )
    assert resumed["request"] == "A local read-heavy workload"
    assert resumed["reviewer"] == "tester"


def test_needs_review_wait_is_a_resumable_recovery_interrupt():
    graph = _one_node_graph("await_recovery", await_recovery)
    config = {"configurable": {"thread_id": "recovery-thread"}}
    paused = graph.invoke({
        "run_id": "run",
        "thread_id": "recovery-thread",
        "stage": "NEEDS_REVIEW",
        "last_error": "The report remains unsupported.",
    }, config)

    assert paused.get("__interrupt__")
    payload = paused["__interrupt__"][0].value
    assert payload["kind"] == "recovery_required"
    assert "retry_report" in payload["allowed_actions"]

    resumed = graph.invoke(
        Command(resume={"action": "retry_report", "additional_attempts": 1}),
        config,
    )
    assert resumed["recovery_decision"]["action"] == "retry_report"
    assert resumed["current_action"] == "Recovery action authorized: retry_report"
