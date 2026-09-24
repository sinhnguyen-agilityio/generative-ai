from intelligence_collector.graph.build import build_graph
def test_graph_foundation_compiles_and_invokes():
    graph=build_graph()
    assert graph.invoke({"run_id": "run", "thread_id": "thread"})["run_id"] == "run"
