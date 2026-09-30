from intelligence_collector.agents.report_judge import (
    build_report_judge_criteria,
    judge_verdict_to_result,
    ReportJudge,
)
from intelligence_collector.contracts import ResearchBrief
from intelligence_collector.models.router import ModelRouter
from tests.fixtures.fake_models import FakeAcceptanceModel

BRIEF = ResearchBrief(research_question="I want a technical report on local cache behavior.")
CRITERIA = ["States how a cache stores data.", "Names at least one primary source."]


def test_verify_sends_brief_criteria_evidence_and_report_and_parses_the_verdict():
    model = FakeAcceptanceModel([{
        "results": [
            {"criterion": CRITERIA[0], "met": True, "justification": "Section explains storage."},
            {"criterion": CRITERIA[1], "met": False, "justification": "No primary source is named."},
        ],
    }])
    judge = ReportJudge(ModelRouter(model))

    verdict = judge.verify(
        BRIEF, CRITERIA, "# Report\n...", "(evidence block)", run_id="run", thread_id="thread",
    )

    assert [r.met for r in verdict.results] == [True, False]
    prompt = model.prompts()[0]
    assert BRIEF.research_question in prompt
    assert CRITERIA[0] in prompt and CRITERIA[1] in prompt
    assert "# Report" in prompt
    assert "(evidence block)" in prompt
    # The judge only says pass/fail and why; it never classifies the failure.
    assert "EVIDENCE_GAP" not in prompt and "DRAFT_DEFECT" not in prompt


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
    assert result.checks == ["criterion_1", "criterion_2"]
    assert result.check_results[0].passed is True


def test_judge_verdict_to_result_fails_an_unmet_criterion_with_its_justification():
    from intelligence_collector.contracts import ReportJudgeVerdict, ReportJudgeCriterion

    verdict = ReportJudgeVerdict(results=[
        ReportJudgeCriterion(criterion=CRITERIA[0], met=True, justification="Satisfied."),
        ReportJudgeCriterion(criterion=CRITERIA[1], met=False, justification="No primary source is named."),
    ])

    result = judge_verdict_to_result(verdict, "hash")

    assert result.passed is False
    assert len(result.check_results) == 2
    failed = [c for c in result.check_results if not c.passed][0]
    assert failed.name == "criterion_2"
    assert "No primary source is named" in failed.details[0]
    assert CRITERIA[1] in failed.details[0]


def test_default_guidelines_match_the_writers_citation_layout():
    from intelligence_collector.agents.prompts import REPORT_WRITER_PROMPT
    from intelligence_collector.config import DEFAULT_REPORT_GUIDELINES

    joined = " ".join(DEFAULT_REPORT_GUIDELINES)
    # The writer cites with [S<n>] markers and keeps URLs only in the Sources list; a guideline
    # demanding a hyperlink in every sub-section (Scope included) could never be satisfied.
    assert "[S1]" in joined and "## Scope" in joined and "## Sources" in joined
    assert "using hyperlinks" not in joined
    assert "Never put URLs anywhere else" in REPORT_WRITER_PROMPT
