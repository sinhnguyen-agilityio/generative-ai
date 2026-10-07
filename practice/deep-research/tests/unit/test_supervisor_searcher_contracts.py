"""Validation rules of the supervisor–searcher contracts (spec 002, data-model.md)."""

import pytest
from pydantic import ValidationError

from intelligence_collector.contracts import (
    ClaimEvaluation,
    EvidencePoint,
    EvidencePointDraft,
    GapAnalysis,
    PointReport,
    ReportStatus,
    ResearchTodo,
    ResultEvaluation,
    SearcherReport,
    StopReason,
    TodoPlan,
    evidence_points_from_drafts,
)


def test_evidence_point_rules():
    point = EvidencePoint(point_id="E1", text=" SQLite size limit ", required_answers=["a", "a"])
    assert point.text == "SQLite size limit" and point.required_answers == ["a"]
    assert point.essential
    assert not EvidencePoint(point_id="E2", text="history").essential
    for bad in (
        {"point_id": "E1", "text": "   "},
        {"point_id": "E1", "text": "x" * 301},
        {"point_id": "E5", "text": "x"},
        {"point_id": "P1", "text": "x"},
    ):
        with pytest.raises(ValidationError):
            EvidencePoint(**bad)


def test_todo_rejects_more_than_four_points_and_duplicate_ids():
    points = [EvidencePoint(point_id=f"E{i}", text=f"p{i}") for i in range(1, 5)]
    assert len(ResearchTodo(research_question="q", evidence_points=points).evidence_points) == 4
    with pytest.raises(ValidationError):
        ResearchTodo(research_question="q", evidence_points=[*points, points[0]])
    with pytest.raises(ValidationError):
        ResearchTodo(research_question="q", evidence_points=[points[0], points[0]])
    with pytest.raises(ValidationError):
        ResearchTodo(research_question="q", round=0)


def test_todo_plan_needs_one_to_four_points_per_question_in_order():
    one = [{"text": "x", "required_answers": [1]}]
    TodoPlan(questions=["a", "b"], evidence_points=[one, one])
    with pytest.raises(ValidationError):
        TodoPlan(questions=["a", "b"], evidence_points=[one])
    with pytest.raises(ValidationError):
        TodoPlan(questions=["a"], evidence_points=[[]])
    with pytest.raises(ValidationError):
        TodoPlan(questions=["a"], evidence_points=[one * 5])
    with pytest.raises(ValidationError):
        TodoPlan(questions=["a"])
    with pytest.raises(ValidationError):
        EvidencePointDraft(text="x", required_answers=[0])


def test_drafts_resolve_to_checklist_texts():
    drafts = [EvidencePointDraft(text="size", required_answers=[2]), EvidencePointDraft(text="misc")]
    points = evidence_points_from_drafts(drafts, ["licence", "size limit"])
    assert [(p.point_id, p.required_answers) for p in points] == [("E1", ["size limit"]), ("E2", [])]
    with pytest.raises(ValueError, match="outside 1..2"):
        evidence_points_from_drafts([EvidencePointDraft(text="x", required_answers=[3])], ["a", "b"])


def test_result_evaluation_reason_rules():
    base = {"url": "https://a", "search_index": 1}
    ResultEvaluation(**base, decision="accepted", covers=["E1"], snippet_only=True)
    ResultEvaluation(**base, decision="rejected", reason="off_topic")
    for bad in (
        {"decision": "rejected"},
        {"decision": "accepted", "reason": "off_topic"},
        {"decision": "rejected", "reason": "duplicate", "covers": ["E1"]},
        {"decision": "rejected", "reason": "duplicate", "snippet_only": True},
    ):
        with pytest.raises(ValidationError):
            ResultEvaluation(**base, **bad)


def test_point_report_and_searcher_report_rules():
    with pytest.raises(ValidationError):
        PointReport(point_id="E1", status="answered")
    PointReport(point_id="E1", status="missing")
    point = PointReport(point_id="E1", status="missing")
    finding = {"text": "f", "source_ids": ["src_a"]}
    report = SearcherReport(
        todo_id="t", points=[point], findings=[finding] * 5, stop_reason=StopReason.NO_PROGRESS
    )
    assert report.report_status == ReportStatus.COMPLETE
    with pytest.raises(ValidationError):
        SearcherReport(todo_id="t", points=[point], findings=[finding] * 6, stop_reason="no_progress")
    with pytest.raises(ValidationError):
        SearcherReport(todo_id="t", points=[point, point], stop_reason="no_progress")
    with pytest.raises(ValidationError):
        SearcherReport(todo_id="t", points=[point], stop_reason="no_progress", corrections=["x"])
    with pytest.raises(ValidationError):
        SearcherReport(todo_id="t", points=[point], stop_reason="report_unreadable")


def test_gap_analysis_and_claim_evaluation_bounds():
    GapAnalysis(round=1, turns_used=4, turn_limit_reached=True, ended="turn_limit")
    with pytest.raises(ValidationError):
        GapAnalysis(round=0)
    with pytest.raises(ValidationError):
        GapAnalysis(round=2, turns_used=5)
    ClaimEvaluation(draft_version=1, artifact_hash="h", supported_share=1.0, passed=True)
    with pytest.raises(ValidationError):
        ClaimEvaluation(draft_version=1, artifact_hash="h", supported_share=1.2, passed=True)
