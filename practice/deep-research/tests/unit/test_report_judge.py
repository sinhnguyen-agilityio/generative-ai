from intelligence_collector.agents.report_judge import (
    build_report_judge_criteria,
    judge_verdict_to_result,
    ReportJudge,
)
from intelligence_collector.contracts import IssueType, ResearchBrief
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.source_verification import EvidenceBundle, EvidenceGap, SourceCatalog, gap_id
from tests.fixtures.fake_models import FakeAcceptanceModel

BRIEF = ResearchBrief(research_question="I want a technical report on local cache behavior.")
CRITERIA = ["States how a cache stores data.", "Names at least one primary source."]


def test_verify_sends_brief_criteria_evidence_gaps_and_report_and_parses_the_verdict():
    model = FakeAcceptanceModel([{
        "results": [
            {"criterion": CRITERIA[0], "met": True, "justification": "Section explains storage."},
            {"criterion": CRITERIA[1], "met": False, "justification": "No primary source is named."},
        ],
    }])
    judge = ReportJudge(ModelRouter(model))
    catalog = SourceCatalog(bundles=[EvidenceBundle(
        todo_id="todo-1", question="How does caching work?", summary="A cache stores data.",
        source_ids=[], gaps=[EvidenceGap(gap_id("todo-1", "eviction policy"), "todo-1",
                                          "How does caching work?", "eviction policy")],
    )])

    verdict = judge.verify(
        BRIEF, CRITERIA, "# Report\n...", "(evidence block)", catalog, run_id="run", thread_id="thread",
    )

    assert [r.met for r in verdict.results] == [True, False]
    prompt = model.prompts()[0]
    assert BRIEF.research_question in prompt
    assert CRITERIA[0] in prompt and CRITERIA[1] in prompt
    assert "# Report" in prompt
    assert "(evidence block)" in prompt
    assert "eviction policy" in prompt  # unresolved gap surfaced to the judge


def test_build_report_judge_criteria_is_just_the_fact_check_by_default():
    criteria = build_report_judge_criteria()

    assert len(criteria) == 1
    assert "inaccuracies" in criteria[0].casefold() and "hallucinations" in criteria[0].casefold()


def test_build_report_judge_criteria_appends_configured_guidelines():
    guidelines = ["The report MUST be written in Markdown format.", "The report MUST be written in English."]

    criteria = build_report_judge_criteria(guidelines)

    assert criteria[1:] == guidelines
    assert len(criteria) == 3


def test_judge_verdict_to_result_passes_only_when_every_criterion_is_met():
    from intelligence_collector.contracts import ReportJudgeVerdict, ReportJudgeCriterion

    verdict = ReportJudgeVerdict(results=[
        ReportJudgeCriterion(criterion=CRITERIA[0], met=True, justification="Satisfied."),
        ReportJudgeCriterion(criterion=CRITERIA[1], met=True, justification="Satisfied."),
    ])

    result = judge_verdict_to_result(verdict, "hash")

    assert result.passed is True
    assert result.issue_type is None
    assert result.checks == ["criterion_1", "criterion_2"]
    assert result.check_results[0].passed is True


def test_judge_verdict_to_result_defaults_an_unmet_criterion_to_draft_defect():
    from intelligence_collector.contracts import ReportJudgeVerdict, ReportJudgeCriterion

    verdict = ReportJudgeVerdict(results=[
        ReportJudgeCriterion(criterion=CRITERIA[0], met=True, justification="Satisfied."),
        ReportJudgeCriterion(criterion=CRITERIA[1], met=False, justification="No primary source is named."),
    ])

    result = judge_verdict_to_result(verdict, "hash")

    assert result.passed is False
    assert result.issue_type == IssueType.DRAFT_DEFECT
    assert len(result.check_results) == 2
    failed = [c for c in result.check_results if not c.passed][0]
    assert failed.name == "criterion_2"
    assert "No primary source is named" in failed.details[0]
    assert CRITERIA[1] in failed.details[0]


def test_judge_verdict_to_result_propagates_evidence_gap_issue_type():
    from intelligence_collector.contracts import ReportJudgeVerdict, ReportJudgeCriterion

    verdict = ReportJudgeVerdict(results=[
        ReportJudgeCriterion(
            criterion=CRITERIA[0], met=False, issue_type=IssueType.EVIDENCE_GAP,
            justification="The evidence never covers this.",
        ),
    ])

    result = judge_verdict_to_result(verdict, "hash")

    assert result.passed is False
    assert result.issue_type == IssueType.EVIDENCE_GAP
