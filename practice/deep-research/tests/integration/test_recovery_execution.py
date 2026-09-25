from intelligence_collector.agents.recovery import apply_repair, persist_review_artifact
from intelligence_collector.agents.planner import replan_affected_node
from intelligence_collector.agents.synthesis import revise_draft_node
from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.agents.worker import targeted_research_node
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP
from tests.fixtures.fake_models import FakeResearchModel
from intelligence_collector.contracts import (
    IndexedChunk,
    IssueType,
    ResearchGraphState,
    ResearchTodo,
    ScopedRequest,
    Stage,
    TodoStatus,
    VerificationResult,
)


def scope():
    return ScopedRequest(
        objective="objective",
        research_boundaries="boundaries",
        constraints="constraints",
        success_criteria="criteria",
        deliverable_format="report",
    )


def failed_verification():
    return VerificationResult(
        artifact_hash="old-draft-hash",
        passed=False,
        issue_type=IssueType.DRAFT_DEFECT,
        affected_ids=["claim-1"],
        repair_details="repair it",
    )


def test_draft_repair_reuses_evidence_and_reenters_verification(stores):
    vfs, _ = stores
    evidence = vfs.write("draft-repair", "research/t/evidence.json", "{}", "evidence_bundle")
    state = ResearchGraphState(
        run_id="draft-repair",
        thread_id="t",
        scoped_request=scope(),
        draft="old draft",
        evidence=[evidence],
        verification=failed_verification(),
    )
    chunks = [IndexedChunk(
        source_id="source-1",
        artifact_id=evidence.artifact_id,
        text="retained evidence",
    )]

    updated = revise_draft_node(state, chunks, "fix claim wording", vfs)

    assert updated.stage == Stage.VERIFYING
    assert updated.verification is None
    assert updated.previous_draft == "old draft"
    assert updated.evidence == [evidence]
    assert "retained evidence" in updated.draft
    assert vfs.read("draft-repair", "reports/previous.md") == b"old draft"


def test_targeted_gap_research_preserves_prior_evidence(stores):
    vfs, index = stores
    prior = vfs.write("gap-repair", "research/old/evidence.json", "{}", "evidence_bundle")
    state = ResearchGraphState(
        run_id="gap-repair",
        thread_id="t",
        scoped_request=scope(),
        evidence=[prior],
        artifacts=[prior],
        verification=failed_verification(),
    )
    mcp = ControlledMCP([{"url": "https://fixture", "content": "target evidence"}])
    researcher = ResearchAgent(ModelRouter(FakeResearchModel(), mode="fake"), lambda: mcp)

    updated = targeted_research_node(state, vfs, index, "target gap", researcher)

    assert updated.stage == Stage.SYNTHESIZING
    assert updated.verification is None
    assert updated.evidence[0] == prior
    assert len(updated.evidence) == 2
    assert updated.evidence[-1].kind == "evidence_bundle"
    assert updated.index_refs
    assert mcp.calls == 1


def test_replan_replaces_only_affected_tasks_and_invalidates_verification(stores):
    vfs, _ = stores
    retained = ResearchTodo(research_question="valid work", status=TodoStatus.COMPLETED)
    affected = ResearchTodo(research_question="too broad", status=TodoStatus.FAILED)
    evidence = vfs.write("plan-repair", "research/valid/evidence.json", "{}", "evidence_bundle")
    state = ResearchGraphState(
        run_id="plan-repair",
        thread_id="t",
        scoped_request=scope(),
        todo_ids=[retained.todo_id, affected.todo_id],
        todos=[retained, affected],
        evidence=[evidence],
        verification=failed_verification(),
    )

    updated = replan_affected_node(
        state,
        vfs,
        [affected.todo_id],
        ["narrow replacement question"],
    )

    assert updated.stage == Stage.RESEARCHING
    assert updated.verification is None
    assert updated.todos[0] == retained
    assert updated.todos[0].status == TodoStatus.COMPLETED
    assert affected.todo_id not in updated.todo_ids
    assert updated.todos[-1].research_question == "narrow replacement question"
    assert updated.todos[-1].status == TodoStatus.PLANNED
    assert updated.evidence == [evidence]
    assert updated.artifacts[-1].kind == "replan"


def test_exhaustion_persists_needs_review(stores):
    vfs, _ = stores
    state = ResearchGraphState(run_id="review", thread_id="t", recovery_attempt=3, recovery_budget=3)
    updated = apply_repair(state, IssueType.DRAFT_DEFECT, feedback="still broken", new_fingerprint="new")
    ref = persist_review_artifact(vfs, updated, "still broken")
    assert updated.status == "needs_review" and ref.kind == "needs_review"
