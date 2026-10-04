"""The reviewer's "extend" reply: stored evidence first, research only what is missing."""

import json

import pytest

from intelligence_collector.agents.planner import plan_record
from intelligence_collector.contracts import (
    ExtensionEvidenceVerdict,
    FeedbackDecision,
    IndexedChunk,
    ResearchBrief,
    ResearchTodo,
    TodoStatus,
)
from intelligence_collector.graph.nodes import ApprovalNodes, BVFNodes, Node
from intelligence_collector.graph.routing import route_after_extension, route_after_feedback
from intelligence_collector.source_verification import load_catalog
from intelligence_collector.storage.vector_index import LocalVectorIndex
from intelligence_collector.storage.vfs import VFS

REQUEST = "Add a section on checkpointing specifically."
TOPIC = "How does LangGraph checkpointing work?"
BRIEF = ResearchBrief(research_question="I want a technical overview of what LangGraph is.",
                      original_question="What is langgraph")
SOURCES = {
    "overview": ("https://docs.test/overview", "LangGraph builds stateful agent graphs."),
    "persistence": ("https://docs.test/persistence",
                    "LangGraph checkpointing saves a snapshot of graph state at every step."),
}


class Judge:
    def __init__(self, verdict=None, error=None):
        self.verdict, self.error, self.calls = verdict, error, []

    def judge(self, question, request, topic, passages, **kwargs):
        self.calls.append([source.source_id for source in passages])
        if self.error:
            raise self.error
        return self.verdict


def _nodes(tmp_path, judge, *, with_chunks=True):
    vfs = VFS(tmp_path / "vfs")
    index = LocalVectorIndex(tmp_path / "index.json")
    nodes = object.__new__(BVFNodes)
    nodes.vfs, nodes.index, nodes.extension_judge = vfs, index, judge
    todo = ResearchTodo(todo_id="t1", research_question="What is LangGraph?",
                        status=TodoStatus.COMPLETED)
    ref = vfs.write_json("r", "research/t1/evidence/result.json", {
        "task_id": "t1", "question": todo.research_question, "summary": "LangGraph overview.",
        "sources": [{"locator": url, "provenance": {"title": name}, "excerpt": text}
                    for name, (url, text) in SOURCES.items()],
        "sufficient": True, "unresolved_gaps": [],
    }, "evidence_bundle", task_id="t1")
    if with_chunks:
        index.index([IndexedChunk(source_id=name, artifact_id=name, text=text,
                                  metadata={"run_id": "r", "todo_id": "t1", "locator": url})
                     for name, (url, text) in SOURCES.items()])
    plan = plan_record(BRIEF, [todo], "overview")
    state = {
        "run_id": "r", "scoped_request": BRIEF.model_dump(mode="json"),
        "todos": [todo.model_dump(mode="json")], "evidence": [ref.model_dump(mode="json")],
        "plan": plan, "plan_version": 1,
        "plan_approval": {"plan_hash": plan["plan_hash"], "approved": True},
        "human_feedback": REQUEST, "reviewer": "tester",
        "extension": {"topic": TOPIC, "request": REQUEST},
    }
    return nodes, state


def test_extend_reply_keeps_the_report_and_goes_to_locate_extension(tmp_path):
    class Router:
        def route(self, *args, **kwargs):
            return FeedbackDecision(route="extend", topic=TOPIC)

    nodes = ApprovalNodes(VFS(tmp_path), str(tmp_path / "exports"), Router())
    result = nodes.classify_feedback({"human_feedback": REQUEST, "draft": "verified report",
                                      "request": "What is langgraph",
                                      "reviewer_requests": ["Use bullets."]})

    assert route_after_feedback(result) == "extend"
    assert result["extension"] == {"topic": TOPIC, "request": REQUEST}
    # Unlike "scope", the question, the report and earlier requests are kept.
    assert result["request"] == "What is langgraph"
    assert result["draft"] == "verified report"
    assert result["reviewer_requests"] == ["Use bullets.", REQUEST]
    assert result["human_feedback"] == REQUEST
    assert result["verification"] is None and result["approval"] is None


def test_extend_without_a_topic_uses_the_reply_as_topic(tmp_path):
    class Router:
        def route(self, *args, **kwargs):
            return FeedbackDecision(route="extend")

    nodes = ApprovalNodes(VFS(tmp_path), str(tmp_path / "exports"), Router())
    result = nodes.classify_feedback({"human_feedback": REQUEST, "draft": "report"})
    assert result["extension"]["topic"] == REQUEST


def test_sufficient_stored_evidence_skips_research(tmp_path):
    judge = Judge(ExtensionEvidenceVerdict(
        sufficient=True, relevant_source_ids=["S2", "S99"],
        summary="Checkpoints snapshot graph state at every step."))
    nodes, state = _nodes(tmp_path, judge)

    result = nodes.locate_extension_evidence(state)

    assert route_after_extension(result) == Node.SYNTHESIZE
    assert judge.calls == [["S2", "S1"]]  # the persistence page ranks first for the topic
    # "S99" was never shown to the judge, so it is dropped.
    assert result["extension"]["relevant_source_ids"] == ["S2"]
    new = ResearchTodo.model_validate(result["todos"][-1])
    assert new.status == TodoStatus.COMPLETED and new.research_question == TOPIC
    assert [todo["todo_id"] for todo in result["todos"]][:1] == ["t1"]
    # The new task's bundle holds only the relevant passage and keeps its citation id.
    catalog = load_catalog(nodes.vfs, "r", result["evidence"],
                           active_todo_ids=[todo["todo_id"] for todo in result["todos"]])
    bundle = next(b for b in catalog.bundles if b.todo_id == new.todo_id)
    assert bundle.source_ids == ["S2"]
    assert bundle.summary == "Checkpoints snapshot graph state at every step."
    assert result["human_feedback"] == REQUEST  # the writer still rewrites the reviewed report


def test_insufficient_evidence_researches_only_the_missing_points(tmp_path):
    judge = Judge(ExtensionEvidenceVerdict(
        sufficient=False, relevant_source_ids=["S2"],
        missing_points=["checkpointer backends", "how a thread resumes"]))
    nodes, state = _nodes(tmp_path, judge)

    result = nodes.locate_extension_evidence(state)

    assert route_after_extension(result) == Node.RESEARCH
    old, new = (ResearchTodo.model_validate(todo) for todo in result["todos"])
    assert old.status == TodoStatus.COMPLETED  # never researched again
    assert new.status == TodoStatus.PLANNED
    assert new.research_instruction == "checkpointer backends; how a thread resumes"
    # The added task is approved by the reviewer's request, so researching accepts the plan.
    assert result["plan"]["kind"] == "extension"
    assert [task["todo_id"] for task in result["plan"]["tasks"]] == ["t1", new.todo_id]
    assert result["plan_approval"]["plan_hash"] == result["plan"]["plan_hash"]
    assert result["plan_approval"]["approved"] and result["plan_approval"]["actor"] == "tester"


@pytest.mark.parametrize("verdict", [
    # Claims sufficiency but cites nothing it was shown.
    ExtensionEvidenceVerdict(sufficient=True, relevant_source_ids=["S99"], summary="x"),
    ExtensionEvidenceVerdict(sufficient=True, relevant_source_ids=[], summary="x"),
])
def test_sufficient_without_a_real_passage_still_researches(tmp_path, verdict):
    nodes, state = _nodes(tmp_path, Judge(verdict))
    result = nodes.locate_extension_evidence(state)
    assert route_after_extension(result) == Node.RESEARCH
    assert result["todos"][-1]["research_instruction"] == ""


def test_judge_failure_falls_back_to_research(tmp_path):
    nodes, state = _nodes(tmp_path, Judge(error=RuntimeError("model call failed")))
    result = nodes.locate_extension_evidence(state)
    assert route_after_extension(result) == Node.RESEARCH
    assert result["extension"]["judge_error"] == "RuntimeError: extension judge failed"


def test_no_stored_passages_researches_without_calling_the_judge(tmp_path):
    judge = Judge(error=AssertionError("must not be called"))
    nodes, state = _nodes(tmp_path, judge, with_chunks=False)
    result = nodes.locate_extension_evidence(state)
    assert route_after_extension(result) == Node.RESEARCH
    assert judge.calls == []
    assert result["extension"]["passages"] == []


def test_extension_record_is_stored(tmp_path):
    nodes, state = _nodes(tmp_path, Judge(ExtensionEvidenceVerdict(
        sufficient=False, missing_points=["checkpointer backends"])))
    result = nodes.locate_extension_evidence(state)
    stored = json.loads(nodes.vfs.read("r", "extension/1.json"))
    assert stored == result["extension"]
    assert result["bvf_history"][-1]["kind"] == "EXTENSION"
