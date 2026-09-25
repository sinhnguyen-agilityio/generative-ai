from intelligence_collector.contracts import ResearchGraphState, Stage
from intelligence_collector.storage.checkpoints import Checkpoints
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.worker import run_worker
from intelligence_collector.contracts import ResearchTodo
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.fake_models import FakeResearchModel

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
    clients = []

    def factory():
        clients.append(ControlledMCP([{"url": "https://fixture", "content": "evidence"}]))
        return clients[-1]

    researcher = ResearchAgent(ModelRouter(FakeResearchModel(), mode="fake"), factory)
    run_worker(vfs, index, "run", todo, researcher)
    run_worker(vfs, index, "run", todo, researcher)
    assert len(clients) == 1  # the second run never starts a sub-agent
