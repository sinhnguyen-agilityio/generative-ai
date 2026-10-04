"""Coverage gate: criteria built from the brief's checklist, id-based matching, one supplemental
call for omitted criteria, ERROR when verdicts stay missing, and mention-is-not-an-answer rules."""

import pytest

from intelligence_collector.agents.report_judge import (
    ReportJudge,
    build_coverage_criteria,
    coverage_checks,
    verification_result,
)
from intelligence_collector.contracts import ReportJudgeVerdict, ResearchBrief
from intelligence_collector.models.router import ModelRouter
from tests.fixtures.fake_models import FakeAcceptanceModel

BRIEF = ResearchBrief(
    research_question="For HTTP 429: which RFC defines it, what does it mean, and which header says when to retry?",
    required_answers=["which RFC defines 429", "what 429 means", "which header says when to retry"],
    output_requirements=["at most 200 words"],
    original_question="For HTTP 429: (1) which RFC, (2) meaning, (3) retry header?",
)
GUIDELINES = ["The report MUST be written in English."]


def test_criteria_list_each_required_answer_then_constraints():
    criteria = build_coverage_criteria(BRIEF, GUIDELINES, ["Use a table"])

    assert [c.label for c in criteria] == [
        "[C1] Answers: which RFC defines 429",
        "[C2] Answers: what 429 means",
        "[C3] Answers: which header says when to retry",
        "[C4] Output requirement: at most 200 words",
        "[C5] The report MUST be written in English.",
        "[C6] Satisfies this reviewer request: Use a table",
    ]
    assert [c.kind for c in criteria] == ["answer"] * 3 + ["output"] * 3


def test_a_brief_without_a_checklist_is_judged_as_one_question():
    brief = ResearchBrief(research_question="What is the latest stable version of Python?")
    assert [c.text for c in build_coverage_criteria(brief)] == [
        "Answers: What is the latest stable version of Python?"]


def test_prompt_requires_an_answer_not_a_mention_and_names_missing_points():
    model = FakeAcceptanceModel([{"results": [{"criterion": "all", "met": True, "justification": "ok"}]}])

    ReportJudge(ModelRouter(model)).assess(
        BRIEF, build_coverage_criteria(BRIEF), "# Report", "(evidence block)")

    prompt = " ".join(model.prompts()[0].split())
    assert "Merely mentioning the topic, describing a source, or saying what a document contains is NOT an answer" in prompt
    assert "if the evidence contains it, the item is not met" in prompt
    assert "missing_points" in prompt
    assert BRIEF.original_question in prompt and "(evidence block)" in prompt


def test_report_that_mentions_but_does_not_answer_fails_with_the_item_and_missing_points():
    model = FakeAcceptanceModel([{"results": [
        {"criterion": "[C1]", "met": True, "justification": "Names RFC 6585."},
        {"criterion": "[C2]", "met": False, "justification": "Describes the RFC but never states the meaning.",
         "missing_points": ["states that the client sent too many requests in a given time"],
         "issue_type": "draft_defect"},
        {"criterion": "[C3]", "met": True, "justification": "Names Retry-After."},
    ]}])
    pairs = ReportJudge(ModelRouter(model)).assess(
        BRIEF, build_coverage_criteria(BRIEF), "# Report", "(evidence block)")

    result = verification_result(coverage_checks(pairs), "hash")

    assert result.passed is False
    failed = [c for c in result.check_results if not c.passed]
    assert len(failed) == 1
    assert failed[0].gate == "coverage" and failed[0].item == "what 429 means"
    assert failed[0].missing_points == ["states that the client sent too many requests in a given time"]
    assert "never states the meaning" in result.repair_details


def test_unanswered_question_is_never_downgraded_to_presentation():
    model = FakeAcceptanceModel([{"results": [
        {"criterion": "[C1]", "met": False, "justification": "Missing.", "issue_type": "presentation"}]}])
    brief = ResearchBrief(research_question="Which RFC defines HTTP status 429?")
    pairs = ReportJudge(ModelRouter(model)).assess(brief, build_coverage_criteria(brief), "r", "e")

    assert coverage_checks(pairs)[0].issue_type == "draft_defect"


class _Router:
    def __init__(self, *verdicts):
        self.verdicts = list(verdicts)
        self.calls = 0

    def invoke_messages(self, *args, **kwargs):
        self.calls += 1
        return ReportJudgeVerdict(results=self.verdicts.pop(0))


def test_verbose_or_duplicated_criteria_are_matched_by_id():
    router = _Router([
        {"criterion": "[C1] Answers: which RFC defines 429 (paraphrased)", "met": True, "justification": "ok"},
        {"criterion": "C1", "met": False, "justification": "duplicate is ignored"},
        {"criterion": "[C2]", "met": True, "justification": "ok"},
        {"criterion": "[C9] invented", "met": False, "justification": "unknown is ignored"},
    ])
    brief = ResearchBrief(research_question="RFC and meaning of 429?", required_answers=["rfc", "meaning"])

    pairs = ReportJudge(router).assess(brief, build_coverage_criteria(brief), "r", "e")

    assert [verdict.met for _, verdict in pairs] == [True, True]
    assert router.calls == 1


def test_omitted_criteria_get_one_supplemental_call():
    router = _Router(
        [{"criterion": "[C1]", "met": True, "justification": "ok"}],
        [{"criterion": "[C2]", "met": False, "justification": "missing meaning"}],
    )
    brief = ResearchBrief(research_question="RFC and meaning of 429?", required_answers=["rfc", "meaning"])

    pairs = ReportJudge(router).assess(brief, build_coverage_criteria(brief), "r", "e")

    assert router.calls == 2
    assert [verdict.met for _, verdict in pairs] == [True, False]


def test_criteria_still_missing_after_the_supplemental_call_are_an_error():
    router = _Router([{"criterion": "[C1]", "met": True, "justification": "ok"}],
                     [{"criterion": "[C1]", "met": True, "justification": "again"}])
    brief = ResearchBrief(research_question="RFC and meaning of 429?", required_answers=["rfc", "meaning"])

    with pytest.raises(ValueError, match="no verdict for C2"):
        ReportJudge(router).assess(brief, build_coverage_criteria(brief), "r", "e")


def test_default_guidelines_match_the_writers_citation_layout():
    from intelligence_collector.agents.prompts import REPORT_WRITER_PROMPT
    from intelligence_collector.config import DEFAULT_REPORT_GUIDELINES

    joined = " ".join(DEFAULT_REPORT_GUIDELINES)
    assert "[S1]" in joined and "## Sources" in joined
    assert "Never put URLs anywhere else" in REPORT_WRITER_PROMPT
