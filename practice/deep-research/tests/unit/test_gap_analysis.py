"""End-of-round gap analysis (spec 002, FR-016 to FR-021; contracts/supervisor-tools.md)."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage

from intelligence_collector.agents.gap_analysis import GapAnalyzer, collect_gaps
from intelligence_collector.contracts import (
    EvidencePoint,
    IndexedChunk,
    ResearchTodo,
    SearcherReport,
    TaskOrigin,
)
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.storage.passage_lookup import PassageLookup
from intelligence_collector.storage.vector_index import LocalVectorIndex

CASE = json.loads(
    (Path(__file__).parents[2] / "evals/supervisor_searcher/gaps.json").read_text(encoding="utf-8")
)["cases"][0]
CHECKLIST = CASE["required_answers"]


def todos():
    result = []
    for task in CASE["tasks"]:
        points = [
            EvidencePoint(
                point_id=p["point_id"],
                text=p["text"],
                required_answers=[CHECKLIST[i - 1] for i in p["required_answers"]],
            )
            for p in task["evidence_points"]
        ]
        result.append(ResearchTodo(todo_id=task["todo_id"], research_question=task["question"],
                                   evidence_points=points))
    return result


def reports():
    return {
        r["todo_id"]: SearcherReport.model_validate({**r, "stop_reason": "no_progress"})
        for r in CASE["reports"]
    }


def index(tmp_path):
    idx = LocalVectorIndex(tmp_path / "index.json")
    idx.index([
        IndexedChunk(chunk_id=f"run:{p['source_id']}:0", source_id=p["source_id"], artifact_id="a",
                     text=p["text"], metadata={"run_id": "run", "todo_id": p["todo_id"],
                                               "locator": p["url"]})
        for p in CASE["pages"]
    ])
    return idx


class SupervisorModel:
    """Scripted supervisor: coverage verdicts, then follow-up tool calls per turn."""

    model_name = "fake-supervisor"

    def __init__(self, coverage, turns=()):
        self.coverage = coverage
        self.turns = list(turns)
        self.turn = 0
        self.follow_up_prompts = []

    def bind_tools(self, tools, tool_choice=None):
        return self

    def invoke(self, messages, config=None):
        system = str(messages[0].content)
        if system.startswith("You write lookup queries"):
            return AIMessage(content=json.dumps({"points": []}))
        if "reviewing the end of a research round" in system:
            return AIMessage(content=json.dumps({"gaps": self.coverage}))
        self.follow_up_prompts.append(messages)
        calls = self.turns[min(self.turn, len(self.turns) - 1)] if self.turns else []
        self.turn += 1
        return AIMessage(content="" if calls else "Nothing more.", tool_calls=[
            {"name": name, "args": args, "id": f"f{self.turn}-{n}"} for n, (name, args) in enumerate(calls)
        ])


def follow_up(question, keys, answers=(2,)):
    return ("CreateFollowUp", {
        "question": question,
        "evidence_points": [{"text": f"evidence for {question}", "required_answers": list(answers)}],
        "gap_point_keys": keys,
        "reason": "still missing",
    })


def analyze(tmp_path, model, *, todo_list=None, round_number=1, previous=None, stall=0, **caps):
    pages = {p["source_id"]: p["text"] for p in CASE["pages"]}
    analyzer = GapAnalyzer(ModelRouter(model), PassageLookup(index(tmp_path)), **caps)
    return analyzer.analyze(
        run_id="run",
        todos=todo_list or todos(),
        reports=reports(),
        checklist=CHECKLIST,
        round_number=round_number,
        previous=previous,
        stall_count=stall,
        allowed_source_ids=set(pages),
        stored_text=lambda sid: pages.get(sid, ""),
    )


COVER_WRITERS = {"point_key": "t2:E1", "covered": True, "source_id": "src_limits",
                 "quote": "there can only be one writer at a time", "justification": "WAL limit"}


def test_classification_comes_only_from_the_plan_links():
    gaps = {g.point_key: g for g in collect_gaps(todos(), reports())}
    expected = CASE["expected"]
    assert sorted(k for k, g in gaps.items() if g.essential) == expected["essential_gaps"]
    assert sorted(k for k, g in gaps.items() if not g.essential) == expected["non_essential_gaps"]
    assert gaps["t1:E2"].resolution == "limitation"


def test_lookup_covers_a_gap_only_with_a_verified_quote_and_follow_ups_close_the_rest(tmp_path):
    bogus = {"point_key": "t2:E2", "covered": True, "source_id": "src_pg",
             "quote": "PostgreSQL allows unlimited concurrent writers", "justification": "?"}
    model = SupervisorModel([COVER_WRITERS, bogus], turns=[[follow_up("PostgreSQL writers", ["t2:E2"])]])

    result = analyze(tmp_path, model)

    gaps = {g.point_key: g for g in result.analysis.gaps}
    assert gaps["t2:E1"].resolution == "covered_by_lookup"
    assert gaps["t2:E1"].covering_source_id == "src_limits"
    assert gaps["t2:E2"].resolution == "follow_up"  # the bogus quote did not verify
    (todo,) = result.follow_ups
    assert todo.origin == TaskOrigin.FOLLOW_UP and todo.round == 2 and todo.gap_point_ids == ["t2:E2"]
    assert todo.evidence_points[0].required_answers == [CHECKLIST[1]]
    assert result.analysis.ended == "follow_ups_created" and result.analysis.follow_ups == [todo.todo_id]
    assert {r.requested_by for r in result.lookups} == {"supervisor"}


def test_follow_up_refusals(tmp_path):
    existing = ResearchTodo(todo_id="f0", research_question="Earlier follow-up",
                            origin=TaskOrigin.FOLLOW_UP, round=2,
                            evidence_points=[EvidencePoint(point_id="E1", text="x",
                                                           required_answers=[CHECKLIST[1]])])
    model = SupervisorModel([COVER_WRITERS], turns=[[
        follow_up("Unknown gap", ["t9:E1"]),
        follow_up("Covered gap", ["t2:E1"]),
        follow_up("Bad link", ["t2:E2"], answers=(7,)),
        follow_up("earlier FOLLOW-UP", ["t2:E2"], answers=(1,)),
        follow_up("Over the per-answer cap", ["t2:E2"], answers=(2,)),
    ]])

    result = analyze(tmp_path, model, todo_list=[*todos(), existing], max_per_answer=1)

    reasons = [r["reason"] for r in result.analysis.refusals]
    assert reasons[0] == "Refused: t9:E1 is unknown or already covered."
    assert reasons[1] == "Refused: t2:E1 is unknown or already covered."
    assert reasons[2].startswith("Refused: evidence points must link to required answers 1..2")
    assert reasons[3] == "Refused: duplicates follow-up f0."
    assert reasons[4] == f"Refused: follow-up limit reached for {CHECKLIST[1]}."
    assert result.follow_ups == []


def test_turn_limit_ends_the_analysis(tmp_path):
    model = SupervisorModel([], turns=[[follow_up("Unknown", ["t9:E9"])]])
    result = analyze(tmp_path, model)
    assert model.turn == 4
    assert result.analysis.turns_used == 4 and result.analysis.turn_limit_reached
    assert result.analysis.ended == "turn_limit"


def test_run_cap_reached_means_no_follow_up_call(tmp_path):
    model = SupervisorModel([], turns=[[follow_up("x", ["t2:E2"])]])
    result = analyze(tmp_path, model, max_per_run=0)
    assert result.analysis.ended == "caps_reached" and model.turn == 0


def test_stall_counts_follow_up_rounds_only(tmp_path):
    previous = {"gaps": [{"point_key": "t2:E1", "essential": True, "resolution": "follow_up"},
                         {"point_key": "t2:E2", "essential": True, "resolution": "follow_up"}]}
    # Round 1 never counts, even when nothing closed.
    first = analyze(tmp_path, SupervisorModel([]), round_number=1, previous=previous)
    assert first.analysis.stall_count == 0
    # Round 2 that closed nothing counts once; a second such round ends research.
    second = analyze(tmp_path, SupervisorModel([]), round_number=2, previous=previous, stall=0)
    assert second.analysis.stall_count == 1
    model = SupervisorModel([], turns=[[follow_up("x", ["t2:E2"])]])
    third = analyze(tmp_path, model, round_number=3, previous=previous, stall=1)
    assert third.analysis.stall_count == 2 and third.analysis.ended == "stalled" and model.turn == 0
    # Closing a gap resets the counter.
    closed = analyze(tmp_path, SupervisorModel([COVER_WRITERS]), round_number=3, previous=previous, stall=1)
    assert closed.analysis.closed_essential == 1 and closed.analysis.stall_count == 0


def test_out_of_scope_gap_is_a_replan_proposal(tmp_path):
    model = SupervisorModel([], turns=[[("ProposeReplan", {"reason": "needs a mobile-only premise",
                                                           "todo_ids": ["t2"]})]])
    result = analyze(tmp_path, model)
    assert result.replan == {"reason": "needs a mobile-only premise", "todo_ids": ["t2"]}
    assert result.follow_ups == []


def test_no_essential_gap_creates_no_follow_up(tmp_path):
    case = json.loads((Path(__file__).parents[2] / "evals/supervisor_searcher/gaps.json")
                      .read_text(encoding="utf-8"))["cases"][1]
    task = case["tasks"][0]
    todo = ResearchTodo(todo_id="t1", research_question=task["question"], evidence_points=[
        EvidencePoint(point_id=p["point_id"], text=p["text"],
                      required_answers=[case["required_answers"][i - 1] for i in p["required_answers"]])
        for p in task["evidence_points"]])
    report = SearcherReport.model_validate({**case["reports"][0], "stop_reason": "all_points_covered"})
    model = SupervisorModel([])
    result = GapAnalyzer(ModelRouter(model), None).analyze(
        run_id="run", todos=[todo], reports={"t1": report}, checklist=case["required_answers"],
        round_number=1, previous=None, stall_count=0, allowed_source_ids=set(),
        stored_text=lambda sid: "",
    )
    assert result.analysis.ended == "no_essential_gaps" and result.follow_ups == []
    assert [g.resolution for g in result.analysis.gaps] == ["limitation"]
    assert model.turn == 0
