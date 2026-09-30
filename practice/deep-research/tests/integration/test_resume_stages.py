from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.worker import run_worker
from intelligence_collector.contracts import ResearchTodo
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.fake_models import FakeResearchModel

def test_completed_worker_is_not_repeated(stores):
    vfs, index = stores
    todo = ResearchTodo(todo_id="task", research_question="question")
    clients = []

    def factory():
        clients.append(ControlledMCP([{"url": "https://fixture", "content": "evidence"}]))
        return clients[-1]

    researcher = ResearchAgent(ModelRouter(FakeResearchModel()), factory)
    run_worker(vfs, index, "run", todo, researcher)
    run_worker(vfs, index, "run", todo, researcher)
    assert len(clients) == 1  # the second run never starts a sub-agent
