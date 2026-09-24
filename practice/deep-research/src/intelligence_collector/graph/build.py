from intelligence_collector.graph.state import GraphState

def build_graph():
    """Build the minimal typed graph foundation; feature nodes are added in later phases."""
    from langgraph.graph import END, START, StateGraph

    def identity(state: GraphState) -> GraphState:
        return state

    graph = StateGraph(GraphState)
    graph.add_node("identity", identity)
    graph.add_edge(START, "identity")
    graph.add_edge("identity", END)
    return graph.compile()
