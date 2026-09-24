from intelligence_collector.agents.recovery import apply_repair, persist_review_artifact
from intelligence_collector.agents.synthesis import revise_draft
from intelligence_collector.agents.worker import ControlledMCP, targeted_research
from intelligence_collector.contracts import IssueType, ResearchGraphState, ScopedRequest, Stage
from intelligence_collector.agents.planner import replan

def scope():
    return ScopedRequest(objective="objective", research_boundaries="boundaries", constraints="constraints", success_criteria="criteria", deliverable_format="report")

def test_repair_routes_execute_and_preserve_evidence(stores):
    vfs, index = stores; state = ResearchGraphState(run_id="r", thread_id="t", scoped_request=scope())
    mcp = ControlledMCP([{"url": "https://fixture", "content": "target evidence"}])
    todo, refs, chunks = targeted_research(vfs, index, "r", "target gap", mcp)
    revised = revise_draft(scope(), chunks, "add support", vfs, "r")
    replanned = replan(vfs, "r", ["replanned question"])
    updated = apply_repair(state, IssueType.EVIDENCE_GAP, feedback="gap", new_fingerprint="new")
    assert todo.status.value == "completed" and refs and chunks and "target evidence" in revised
    assert replanned[0].research_question == "replanned question"
    assert updated.stage == Stage.RESEARCHING and updated.recovery_attempt == 1

def test_exhaustion_persists_needs_review(stores):
    vfs, _ = stores
    state = ResearchGraphState(run_id="review", thread_id="t", recovery_attempt=3, recovery_budget=3)
    updated = apply_repair(state, IssueType.DRAFT_DEFECT, feedback="still broken", new_fingerprint="new")
    ref = persist_review_artifact(vfs, updated, "still broken")
    assert updated.status == "needs_review" and ref.kind == "needs_review"
