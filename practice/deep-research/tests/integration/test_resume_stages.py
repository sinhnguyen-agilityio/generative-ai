from intelligence_collector.contracts import ResearchGraphState, Stage
from intelligence_collector.storage.checkpoints import Checkpoints
from intelligence_collector.agents.worker import ControlledMCP, run_worker
from intelligence_collector.contracts import ResearchTodo

def test_checkpoint_restores_typed_state_at_each_stage(tmp_path):
    checkpoints = Checkpoints(tmp_path / "checkpoints.json")
    for stage in Stage:
        state = ResearchGraphState(run_id="r", thread_id=stage.value, stage=stage)
        checkpoints.save_state(state)
        restored = checkpoints.load_state(stage.value, ResearchGraphState)
        assert restored.stage == stage

def test_completed_worker_is_not_repeated(stores):
    vfs, index = stores
    todo = ResearchTodo(todo_id="task", research_question="question")
    mcp = ControlledMCP([{"url": "https://fixture", "content": "evidence"}])
    run_worker(vfs, index, "run", todo, mcp)
    calls = mcp.calls
    run_worker(vfs, index, "run", todo, mcp)
    assert mcp.calls == calls
