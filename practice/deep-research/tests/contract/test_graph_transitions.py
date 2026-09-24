from intelligence_collector.graph.build import build_graph
def test_mvp_graph_runs_all_stages():
    result = build_graph().invoke({"run_id": "run", "thread_id": "thread"})
    assert result["stage"] == "verifying"
