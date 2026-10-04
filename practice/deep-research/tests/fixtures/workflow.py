"""Deterministic collaborators; no credentials or live calls."""
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from intelligence_collector.agents.researcher import ResearchOutcome
from intelligence_collector.contracts import ResearchFindings, ReportJudgeCriterion, ReportJudgeVerdict
from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.nodes import ResearchNodes
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.storage.vector_index import LocalVectorIndex

URL = "https://controlled.test/cache"
QUESTION = "How does a local software cache store and reuse data?"
BRIEF = {"research_question": "Explain local software cache behavior using primary sources.",
         "original_questions": [QUESTION]}

class Writer:
    router = None
    def __init__(self):
        self.calls = []
        self.fail = False
    def write(self, brief, catalog, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise ValueError("drafting failed")
        return (f"# Cache report version {len(self.calls)}\n\n## Findings\n"
                f"A cache stores reusable data for later access [S1].\n\n## Sources\n[S1] {URL}")

class Judge:
    def __init__(self, failures=0):
        self.failures, self.calls = failures, []
    def verify(self, brief, criteria, report, evidence, **kwargs):
        self.calls.append(report)
        failed = len(self.calls) <= self.failures
        return ReportJudgeVerdict(results=[
            ReportJudgeCriterion(criterion=c, met=not (failed and i == 0),
                justification="Missing explanation of cache reuse" if failed and i == 0 else "Covered",
                issue_type="draft_defect") for i, c in enumerate(criteria)])

class Researcher:
    router = None
    def __init__(self):
        self.calls = []
    def research(self, question, **kwargs):
        self.calls.append(question)
        return ResearchOutcome(question=question, search_budget=5, mcp_server="controlled",
            mcp_tool="search", search_calls=1, sources=[{"url": URL, "content": "A cache stores reusable data for later access."}],
            findings=ResearchFindings(summary="Cache evidence", sufficient=True))

class Diagnosis:
    def __init__(self, route="vfs"):
        self.route, self.calls = route, []
    def diagnose(self, state, catalog):
        self.calls.append(state.get("draft"))
        return self.route, [{"question": "todo-cache", "location": "Findings",
            "reason": "Repair cache reuse using identified original evidence",
            "evidence_refs": ["S1"] if self.route != "research" else [], "route": self.route}]

def workflow(tmp_path, *, failures=0, score=None, route="vfs", scope_agent=None, planner_agent=None):
    vfs = VFS(tmp_path / "vfs")
    index = LocalVectorIndex(tmp_path / "index.json")
    writer, judge, researcher, diagnosis = Writer(), Judge(failures), Researcher(), Diagnosis(route)
    nodes = ResearchNodes(vfs, index, researcher, report_writer=writer, report_judge=judge,
                          faithfulness_evaluator=score or (lambda *args: .99))
    nodes.bvf.diagnosis = diagnosis
    connection = sqlite3.connect(tmp_path / "checkpoints.sqlite", check_same_thread=False)
    graph = build_graph(research_nodes=nodes, scope_agent=scope_agent, planner_agent=planner_agent,
                        vfs=vfs, with_approval=True, checkpointer=SqliteSaver(connection),
                        output_dir=str(tmp_path / "exports"))
    config = {"configurable": {"thread_id": "run"}, "recursion_limit": 100}
    initial = {"run_id": "run", "thread_id": "run", "request": QUESTION, "scoped_request": BRIEF,
               "todos": [{"todo_id": "todo-cache", "research_question": QUESTION, "status": "planned"}]}
    return graph, config, initial, nodes, writer, judge, researcher, diagnosis

def approve_plan(graph, config, initial):
    paused = graph.invoke(initial, config)
    assert paused["stage"] == "Awaiting Plan Approval"
    assert paused.get("__interrupt__")
    return graph.invoke(Command(resume={"approved": True}), config)
