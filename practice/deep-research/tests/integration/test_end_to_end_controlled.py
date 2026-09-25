from intelligence_collector.agents.planner import create_todos
from intelligence_collector.agents.scope import build_scope
from intelligence_collector.agents.supervisor import assess
from intelligence_collector.agents.synthesis import synthesize
from intelligence_collector.agents.verification import verify
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.worker import run_worker
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.fake_models import FakeResearchModel

def test_controlled_end_to_end(stores):
    vfs, index = stores; run_id = "e2e"
    scope = build_scope({k: k for k in ("objective", "research_boundaries", "constraints", "success_criteria", "deliverable_format")})
    todos = create_todos(vfs, run_id, ["What is a cache?"])
    calls = []
    mcp = ControlledMCP([{"url": "https://controlled.test/cache", "content": "A cache stores reusable data."}])
    original = mcp.search
    def tracked(query, **kwargs):
        assert vfs.read(run_id, "planning/todos.v1.json")
        calls.append(query); return original(query, **kwargs)
    mcp.search = tracked
    researcher = ResearchAgent(ModelRouter(FakeResearchModel(), mode="fake"), lambda: mcp)
    todo, refs, chunks = run_worker(vfs, index, run_id, todos[0], researcher)
    assert calls and todo.status.value == "completed"
    assessment = assess(todos, refs, [c.chunk_id for c in chunks])
    assert assessment.ready_for_synthesis
    draft = synthesize(scope, chunks)
    result = verify(draft, scope, chunks)
    assert result.passed
