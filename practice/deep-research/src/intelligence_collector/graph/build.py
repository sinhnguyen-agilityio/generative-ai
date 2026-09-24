from collections.abc import Callable

from intelligence_collector.graph.state import GraphState


def node_wrapper(name: str, fn: Callable[[GraphState], GraphState]):
    """Normalize node execution so every graph node returns a serializable state update."""
    def wrapped(state: GraphState) -> GraphState:
        result = fn(state)
        if not isinstance(result, dict):
            raise TypeError(f"{name} must return a mapping state update")
        return result
    wrapped.__name__ = name
    return wrapped

def build_graph(*, checkpointer=None, with_approval: bool = False):
    """Build the typed MVP graph with optional resumable approval interrupt."""
    from langgraph.graph import END, START, StateGraph

    def stage(name):
        def advance(state: GraphState) -> GraphState:
            return {**state, "stage": name}
        return node_wrapper(name, advance)

    def approval(state: GraphState) -> GraphState:
        from langgraph.types import interrupt
        decision = interrupt({"kind": "publication_approval", "run_id": state.get("run_id")})
        return {**state, "approval": decision, "stage": "Published" if decision.get("approved") else "Awaiting Approval"}

    graph = StateGraph(GraphState)
    for name in ("scope", "planning", "researching", "synthesizing", "verifying"):
        graph.add_node(name, stage(name))
    graph.add_edge(START, "scope")
    graph.add_edge("scope", "planning")
    graph.add_edge("planning", "researching")
    graph.add_edge("researching", "synthesizing")
    graph.add_edge("synthesizing", "verifying")
    if with_approval:
        graph.add_node("approval", node_wrapper("approval", approval))
        graph.add_edge("verifying", "approval")
        graph.add_edge("approval", END)
    else:
        graph.add_edge("verifying", END)
    return graph.compile(checkpointer=checkpointer)
