from intelligence_collector.agents.acceptance_verifier import AcceptanceVerifier, merge_acceptance_verdict
from intelligence_collector.contracts import IssueType, ResearchBrief, VerificationResult
from intelligence_collector.models.router import ModelRouter
from tests.fixtures.fake_models import FakeAcceptanceModel

BRIEF = ResearchBrief(research_question="I want a technical report on local cache behavior.")
CRITERIA = ["States how a cache stores data.", "Names at least one primary source."]


def _passing_result() -> VerificationResult:
    return VerificationResult(artifact_hash="hash", passed=True, checks=["structure"])


def test_verify_sends_brief_criteria_and_report_and_parses_the_verdict():
    model = FakeAcceptanceModel([{
        "results": [
            {"criterion": CRITERIA[0], "met": True, "justification": "Section explains storage."},
            {"criterion": CRITERIA[1], "met": False, "justification": "No primary source is named."},
        ],
    }])
    verifier = AcceptanceVerifier(ModelRouter(model, mode="fake"))

    verdict = verifier.verify(BRIEF, CRITERIA, "# Report\n...", run_id="run", thread_id="thread")

    assert [r.met for r in verdict.results] == [True, False]
    prompt = model.prompts()[0]
    assert BRIEF.research_question in prompt
    assert CRITERIA[0] in prompt and CRITERIA[1] in prompt
    assert "# Report" in prompt


def test_merge_acceptance_verdict_leaves_a_passing_result_untouched():
    model = FakeAcceptanceModel([{"results": [
        {"criterion": c, "met": True, "justification": "Satisfied."} for c in CRITERIA
    ]}])
    verdict = AcceptanceVerifier(ModelRouter(model, mode="fake")).verify(BRIEF, CRITERIA, "report")

    merged = merge_acceptance_verdict(_passing_result(), verdict)

    assert merged.passed is True
    assert merged.checks == ["structure"]
    assert merged.check_results == []


def test_merge_acceptance_verdict_turns_a_pass_into_a_draft_defect_on_any_unmet_criterion():
    model = FakeAcceptanceModel([{
        "results": [
            {"criterion": CRITERIA[0], "met": True, "justification": "Satisfied."},
            {"criterion": CRITERIA[1], "met": False, "justification": "No primary source is named."},
        ],
    }])
    verdict = AcceptanceVerifier(ModelRouter(model, mode="fake")).verify(BRIEF, CRITERIA, "report")

    merged = merge_acceptance_verdict(_passing_result(), verdict)

    assert merged.passed is False
    assert merged.issue_type == IssueType.DRAFT_DEFECT
    assert merged.checks == ["structure", "acceptance_criteria"]
    assert len(merged.check_results) == 1
    check = merged.check_results[0]
    assert check.name == "acceptance_criteria"
    assert check.passed is False
    assert "No primary source is named" in check.details[0]
    assert CRITERIA[1] in check.details[0]
    assert CRITERIA[0] not in "\n".join(check.details)  # only the unmet criterion is reported
