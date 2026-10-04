from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from intelligence_collector.graph.build import build_graph


def test_graph_approval_interrupt_resumes_by_thread_id():
    graph = build_graph(checkpointer=MemorySaver(), with_approval=True)
    config = {"configurable": {"thread_id": "approval-thread"}}
    paused = graph.invoke({"run_id": "run", "thread_id": "approval-thread"}, config)
    assert paused.get("__interrupt__")
    resumed = graph.invoke(Command(resume={"approved": True}), config)
    assert resumed["stage"] == "Published"
