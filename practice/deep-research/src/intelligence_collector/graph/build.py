from intelligence_collector.graph.state import GraphState

def build_graph():
    """Build the MVP stage graph; domain work is supplied through state payloads."""
    from langgraph.graph import END, START, StateGraph

    def stage(name):
        def advance(state: GraphState) -> GraphState:
            return {**state, "stage": name}
        return advance

    graph = StateGraph(GraphState)
    for name in ("scope", "planning", "researching", "synthesizing", "verifying"):
        graph.add_node(name, stage(name))
    graph.add_edge(START, "scope")
    graph.add_edge("scope", "planning")
    graph.add_edge("planning", "researching")
    graph.add_edge("researching", "synthesizing")
    graph.add_edge("synthesizing", "verifying")
    graph.add_edge("verifying", END)
    return graph.compile()
