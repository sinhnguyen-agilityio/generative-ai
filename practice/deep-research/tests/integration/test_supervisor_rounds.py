"""Research rounds: searchers report, the supervisor analyzes gaps after the whole round and
creates bounded follow-up tasks (spec 002, US3; quickstart scenarios 7–9)."""

import json
from pathlib import Path
import re

from langchain_core.messages import AIMessage

from intelligence_collector.agents.report_judge import ReportJudge
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.contracts import EvidencePoint, ResearchTodo, TaskOrigin
from intelligence_collector.graph.nodes import Node, ResearchNodes
from intelligence_collector.graph.routing import route_after_gap_analysis
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.fake_models import FakeResearchModel

CHECKLIST = ["SQLite maximum database size", "SQLite concurrent writers"]
BRIEF = {"research_question": "Compare SQLite limits for an offline mobile app.",
         "required_answers": CHECKLIST}
_GAP = re.compile(r"^- (t[\w-]*:E\d+): ", re.M)


class RoundModel(FakeResearchModel):
    """Searchers, supervisor delegation (from FakeResearchModel) plus gap analysis: no stored
    passage covers a gap unless `covers` names it; each round, one follow-up per open gap until
    `follow_up_rounds` rounds have created them."""

    def __init__(self, *, covers=None, follow_up_rounds=1, **kwargs):
        super().__init__(**kwargs)
        self.covers = covers or {}
        self.follow_up_rounds = follow_up_rounds
        self.follow_up_turns = []
        self.analyses = 0
        self.reports = 0

    def invoke(self, messages, config=None):
        system = str(messages[0].content)
        if system.startswith("You write the short structured report"):
            self.reports += 1
        if "reviewing the end of a research round" in system:
            self.analyses += 1
            keys = re.findall(r"^(t[\w-]*:E\d+): ", str(messages[-1].content), re.M)
            return AIMessage(content=json.dumps({"gaps": [
                {"point_key": k, "covered": k in self.covers, "source_id": self.covers.get(k, ("", ""))[0],
                 "quote": self.covers.get(k, ("", ""))[1], "justification": "judged"} for k in keys
            ]}))
        if system.startswith("You are the research supervisor. Research round"):
            round_number = int(re.search(r"Research round (\d+)", system).group(1))
            already = any(isinstance(m, AIMessage) for m in messages)
            self.follow_up_turns.append(round_number)
            if already or round_number > self.follow_up_rounds:
                return AIMessage(content="Nothing more to research.")
            keys = _GAP.findall(str(messages[-1].content))
            return AIMessage(content="", tool_calls=[
                {"name": "CreateFollowUp", "id": f"fu-{round_number}-{n}", "args": {
                    "question": f"Round {round_number} follow-up for {key}",
                    "evidence_points": [{"text": f"answer for {key}", "required_answers": [2]}],
                    "gap_point_keys": [key], "reason": "missing"}}
                for n, key in enumerate(keys)
            ])
        return super().invoke(messages, config)


def todo(todo_id, question, answers):
    return ResearchTodo(todo_id=todo_id, research_question=question, evidence_points=[
        EvidencePoint(point_id="E1", text=question, required_answers=answers)
    ])


def make_nodes(stores, model, mcp_factory, **kwargs):
    vfs, index = stores
    researcher = ResearchAgent(ModelRouter(model), mcp_factory)
    return ResearchNodes(vfs, index, researcher, report_judge=ReportJudge(ModelRouter(model)), **kwargs)


def pages():
    """Each searcher's client returns one new page."""
    made = []

    def factory():
        n = len(made)
        made.append(ControlledMCP([[{"url": f"https://p.test/{n}", "content": f"Page {n} facts."}]]))
        return made[-1]

    return factory, made


def run_rounds(nodes, state, limit=6):
    """researching → analyze_gaps until research ends; returns the final state and routes."""
    routes = []
    for _ in range(limit):
        state.update(nodes.researching(state))
        state.update(nodes.analyze_gaps(state))
        routes.append(route_after_gap_analysis(state))
        if routes[-1] != Node.RESEARCH:
            break
    return state, routes


def initial_state(*todos):
    return {"run_id": "run", "thread_id": "thread", "scoped_request": BRIEF,
            "todos": [t.model_dump(mode="json") for t in todos]}


def test_follow_up_is_created_after_the_whole_round_and_closes_the_gap(stores):
    vfs, _ = stores
    t1 = todo("t1", "SQLite maximum database size", [CHECKLIST[0]])
    t2 = todo("t2", "SQLite concurrent writers", [CHECKLIST[1]])
    model = RoundModel(findings={t2.research_question: {"sufficient": False}})
    factory, clients = pages()
    nodes = make_nodes(stores, model, factory)

    state, routes = run_rounds(nodes, initial_state(t1, t2))

    assert routes == [Node.RESEARCH, Node.ASSESS_RESEARCH]
    # Round 1 researched both tasks before any follow-up was created; round 2 only the follow-up.
    assert len(clients) == 3 and model.follow_up_turns[0] == 1
    follow_ups = [ResearchTodo.model_validate(t) for t in state["todos"] if t["origin"] == "follow_up"]
    assert [(f.round, f.gap_point_ids) for f in follow_ups] == [(2, ["t2:E1"])]
    assert state["research_round"] == 2 and state["follow_up_count"] == 1
    assert state["gap_analysis"]["ended"] == "no_essential_gaps"
    round1 = json.loads(vfs.read("run", "research/rounds/round-1.json"))
    assert round1["analysis"]["ended"] == "follow_ups_created"
    state.update(nodes.research_complete(state))
    assert state["research_status"] == "READY"


def test_a_gap_answered_on_another_tasks_page_creates_no_follow_up(stores):
    t1 = todo("t1", "SQLite maximum database size", [CHECKLIST[0]])
    t2 = todo("t2", "SQLite concurrent writers", [CHECKLIST[1]])
    page_text = "SQLite WAL mode: there can only be one writer at a time."
    model = RoundModel(findings={t2.research_question: {"sufficient": False}},
                       covers={"t2:E1": ("src_placeholder", "there can only be one writer at a time")})
    made = []

    def factory():
        n = len(made)
        rows = [{"url": "https://sqlite.test/wal", "content": page_text}] if n == 0 else []
        made.append(ControlledMCP([rows]))
        return made[-1]

    nodes = make_nodes(stores, model, factory)
    # The supervisor's verdict names the real stored source of t1's page.
    from intelligence_collector.tools.page_content import source_id_for

    model.covers["t2:E1"] = (source_id_for("https://sqlite.test/wal"), model.covers["t2:E1"][1])
    state, routes = run_rounds(nodes, initial_state(t1, t2))

    assert routes == [Node.ASSESS_RESEARCH]
    gap = next(g for g in state["gap_analysis"]["gaps"] if g["point_key"] == "t2:E1")
    assert gap["resolution"] == "covered_by_lookup"
    assert state["follow_up_count"] == 0


def test_caps_and_two_rounds_without_progress_end_research(stores):
    t2 = todo("t2", "SQLite concurrent writers", [CHECKLIST[1]])
    # Every searcher reports nothing; the supervisor keeps creating follow-ups when allowed.
    model = RoundModel(follow_up_rounds=10)
    nodes = make_nodes(stores, model, lambda: ControlledMCP([[]]), max_follow_ups_per_answer=2)

    state, routes = run_rounds(nodes, initial_state(t2))

    # Round 1 → F1, round 2 (no progress: stall 1) → F2, round 3: per-answer cap of 2 reached.
    assert state["follow_up_count"] == 2
    assert routes[-1] == Node.ASSESS_RESEARCH
    assert state["gap_analysis"]["ended"] in {"caps_reached", "stalled"}
    assert len(routes) <= 3


def test_stall_stops_follow_ups_after_two_rounds_without_progress(stores):
    t2 = todo("t2", "SQLite concurrent writers", [CHECKLIST[1]])
    model = RoundModel(follow_up_rounds=10)
    nodes = make_nodes(stores, model, lambda: ControlledMCP([[]]),
                       max_follow_ups_per_answer=4, max_follow_ups_per_run=8)

    state, routes = run_rounds(nodes, initial_state(t2))

    # Round 1 never counts; rounds 2 and 3 close nothing, so round 3's analysis ends research.
    assert state["gap_analysis"]["ended"] == "stalled"
    assert state["gap_stall_count"] == 2 and state["follow_up_count"] == 2
    assert routes == [Node.RESEARCH, Node.RESEARCH, Node.ASSESS_RESEARCH]


def test_mcp_failure_is_a_technical_gap_that_the_supervisor_follows_up(stores):
    vfs, _ = stores

    class Down(ControlledMCP):
        def search(self, query, **kwargs):
            raise TimeoutError("GET https://mcp.test/?tavilyApiKey=secret-key timed out")

    t2 = todo("t2", "SQLite concurrent writers", [CHECKLIST[1]])
    model = RoundModel(queries={t2.research_question: [t2.research_question, "sqlite writer lock"]})
    made = []

    def factory():
        made.append(Down([[]]) if not made else ControlledMCP([[{"url": "https://ok.test", "content": "Writers: one at a time."}]]))
        return made[-1]

    nodes = make_nodes(stores, model, factory)
    state, routes = run_rounds(nodes, initial_state(t2))

    report = json.loads(vfs.read("run", "research/t2/report.json"))
    assert report["stop_reason"] == "search_failures" and report["failure_kind"] == "technical"
    assert "secret-key" not in json.dumps(report)
    assert routes[0] == Node.RESEARCH  # the essential point became a gap with a follow-up
    follow_up = next(t for t in state["todos"] if t["origin"] == TaskOrigin.FOLLOW_UP.value)
    assert follow_up["gap_point_ids"] == ["t2:E1"]


def test_follow_ups_do_not_break_the_plan_approval_guard(stores):
    from intelligence_collector.agents.planner import plan_record
    from intelligence_collector.contracts import ResearchBrief

    t1 = todo("t1", "SQLite maximum database size", [CHECKLIST[0]])
    t2 = todo("t2", "SQLite concurrent writers", [CHECKLIST[1]])
    plan = plan_record(ResearchBrief.model_validate(BRIEF), [t1, t2], "short")
    model = RoundModel(findings={t2.research_question: {"sufficient": False}})
    factory, _ = pages()
    nodes = make_nodes(stores, model, factory)
    state = {**initial_state(t1, t2), "plan": plan,
             "plan_approval": {"approved": True, "plan_hash": plan["plan_hash"]}}

    state, routes = run_rounds(nodes, state)

    assert routes[0] == Node.RESEARCH  # round 2 passed the approval check in researching
    assert state["plan"]["plan_hash"] == plan["plan_hash"]
    assert [f["gap_point_ids"] for f in state["plan"]["follow_ups"]] == [["t2:E1"]]


def test_resume_repeats_no_search_report_or_round_analysis(stores):
    vfs, _ = stores
    t1 = todo("t1", "SQLite maximum database size", [CHECKLIST[0]])
    t2 = todo("t2", "SQLite concurrent writers", [CHECKLIST[1]])
    model = RoundModel(findings={t.research_question: {"sufficient": False} for t in (t1, t2)})
    factory, clients = pages()
    nodes = make_nodes(stores, model, factory)

    def spent():
        return len(clients), model.reports, model.analyses, len(model.follow_up_turns)

    state = initial_state(t1, t2)
    state.update(nodes.researching(state))
    before_analysis = dict(state)
    state.update(nodes.analyze_gaps(state))
    after_round_1 = dict(state)
    assert after_round_1["follow_up_count"] == 2 and after_round_1["research_round"] == 2

    # Interrupted after round 1's analysis: the node runs again from the state before it.
    calls = spent()
    resumed = {**before_analysis, **nodes.analyze_gaps(dict(before_analysis))}
    assert spent() == calls  # the saved round file is reused; no model call
    for key in ("todos", "research_round", "follow_up_count", "gap_stall_count", "gap_analysis"):
        assert resumed[key] == after_round_1[key], key

    # Interrupted halfway through round 2: one follow-up finished, the other had searched and
    # reported but not yet written its evidence bundle.
    before_round_2 = dict(resumed)
    state = {**before_round_2, **nodes.researching(dict(before_round_2))}
    follow_ups = [t["todo_id"] for t in state["todos"] if t["origin"] == "follow_up"]
    unfinished = follow_ups[1]
    (Path(vfs.root) / "run" / f"research/{unfinished}/evidence/result.json").unlink()
    calls = spent()
    resumed = {**before_round_2, **nodes.researching(dict(before_round_2))}
    assert spent() == calls  # no search, capture or report is repeated
    assert vfs.read("run", f"research/{unfinished}/evidence/result.json")
    for key in ("research_round", "follow_up_count", "gap_stall_count"):
        assert resumed[key] == before_round_2[key], key
