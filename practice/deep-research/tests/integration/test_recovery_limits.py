from intelligence_collector.agents.recovery import choose_repair
from intelligence_collector.contracts import IssueType, ResearchGraphState, Stage

def test_recovery_routes_and_budget():
    state = ResearchGraphState(run_id="r", thread_id="t", recovery_budget=1)
    first = choose_repair(state, IssueType.EVIDENCE_GAP, "new")
    assert first.stage == Stage.RESEARCHING and not first.exhausted
    exhausted = choose_repair(state.model_copy(update={"recovery_attempt": 1}), IssueType.EVIDENCE_GAP, "new")
    assert exhausted.stage == Stage.NEEDS_REVIEW and exhausted.exhausted

def test_no_progress_is_reviewed():
    state = ResearchGraphState(run_id="r", thread_id="t", progress_fingerprint="same")
    result = choose_repair(state, IssueType.DRAFT_DEFECT, "same")
    assert result.stage == Stage.NEEDS_REVIEW
