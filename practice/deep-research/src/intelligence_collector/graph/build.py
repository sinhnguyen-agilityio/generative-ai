from intelligence_collector.contracts import ResearchGraphState, Stage
def build_graph():
    """Return a LangGraph graph when installed; the domain runner remains importable locally."""
    try:
        from langgraph.graph import StateGraph
        return StateGraph(dict)
    except ImportError:
        return None

