"""Repair diagnosis picks the smallest useful correction; the Chroma route reads located evidence
back from VFS provenance; plan decisions and URL-keyed artifacts behave as documented."""

import json

import pytest

from intelligence_collector.agents.worker import persist_research
from intelligence_collector.agents.researcher import ResearchOutcome
from intelligence_collector.contracts import (
    PlanApprovalDecision,
    ResearchFindings,
    ResearchTodo,
    VerificationCheck,
    VerificationResult,
)
from intelligence_collector.repair import diagnose, item_key, locate_with_index
from intelligence_collector.source_verification import EvidenceBundle, SourceCatalog, load_catalog
from intelligence_collector.storage.vfs import VFS

TODOS = [ResearchTodo(todo_id="a", research_question="What does HTTP 401 mean?"),
         ResearchTodo(todo_id="b", research_question="What does HTTP 403 mean?")]


def _catalog(source_ids=("S1", "S2", "S3")):
    return SourceCatalog(bundles=[
        EvidenceBundle("a", TODOS[0].research_question, "", ["S1"], []),
        EvidenceBundle("b", TODOS[1].research_question, "", list(source_ids), []),
    ])


def _failed(*checks):
    return VerificationResult(artifact_hash="h", passed=False, check_results=list(checks))


def _coverage(item, issue, missing=("states the meaning",), passed=False):
    return VerificationCheck(name="coverage_C1", passed=passed, gate="coverage", issue_type=issue,
                             item=item, missing_points=list(missing), details=["detail"])


def _diagnose(*checks, shown=None, researched=(), can_replan=True, catalog=None):
    return diagnose(_failed(*checks), todos=TODOS, full_catalog=catalog or _catalog(),
                    shown_source_ids=shown or {"a": ["S1"], "b": ["S1", "S2", "S3"]},
                    researched_items=list(researched), can_replan=can_replan)


def test_answer_present_in_stored_evidence_is_rewritten_from_vfs():
    plan = _diagnose(_coverage("what 403 means", "draft_defect"))
    assert plan["route"] == "vfs"
    assert plan["issues"][0]["todo_id"] == "b"  # mapped to the task about 403


def test_unseen_stored_sources_are_located_with_chroma_before_any_new_research():
    plan = _diagnose(_coverage("what 403 means", "evidence_gap"), shown={"b": ["S1"]})
    assert plan["route"] == "chroma"


def test_missing_evidence_gets_targeted_research_then_a_replacement_plan():
    gap = _coverage("what 403 means", "evidence_gap")
    assert _diagnose(gap)["route"] == "research"
    assert _diagnose(gap, researched=[item_key("what 403 means")])["route"] == "replan"
    assert _diagnose(gap, researched=[item_key("what 403 means")], can_replan=False)["route"] == "vfs"


def test_presentation_and_faithfulness_routes_and_priority():
    presentation = VerificationCheck(name="coverage_C4", passed=False, gate="coverage",
                                     issue_type="presentation", item="English", details=["French"])
    unsupported = VerificationCheck(name="faithfulness", passed=False, gate="faithfulness",
                                    missing_points=["Invented claim."], details=["0.80"])
    missing_url = VerificationCheck(name="citation_resolution", passed=False, gate="citation",
                                    issue_type="presentation", details=["Missing Sources section"])
    assert _diagnose(presentation)["route"] == "presentation"
    assert _diagnose(missing_url)["route"] == "presentation"
    assert _diagnose(unsupported)["route"] == "vfs"
    mixed = _diagnose(presentation, _coverage("what 403 means", "evidence_gap"))
    assert mixed["route"] == "research"  # the largest needed route; every issue still listed
    assert [issue["route"] for issue in mixed["issues"]] == ["presentation", "research"]


def test_passing_answers_are_listed_for_preservation():
    ok = _coverage("what 401 means", "draft_defect", missing=(), passed=True)
    plan = _diagnose(ok, _coverage("what 403 means", "draft_defect"))
    assert plan["passing"] == ["what 401 means"]


def test_chroma_route_reads_located_evidence_back_through_vfs_provenance(tmp_path):
    from intelligence_collector.storage.vector_index import LocalVectorIndex

    vfs, index = VFS(tmp_path / "vfs"), LocalVectorIndex(tmp_path / "index.json")
    todo = ResearchTodo(todo_id="b", research_question="What does HTTP 403 mean?")
    rows = [{"url": f"https://x.test/{n}", "title": f"page {n}",
             "content": "unrelated caching notes" if n else "403 Forbidden means the server refuses to authorize"}
            for n in range(4)]
    outcome = ResearchOutcome(question=todo.research_question, search_budget=1, mcp_server="tavily",
                              mcp_tool="tavily_search", search_calls=1, sources=rows,
                              findings=ResearchFindings(summary="notes", sufficient=True))
    todo, refs, _ = persist_research(vfs, index, "run", todo, outcome)
    artifacts = [ref.model_dump(mode="json") for ref in refs]
    evidence = [ref.model_dump(mode="json") for ref in refs if ref.kind == "evidence_bundle"]
    full = load_catalog(vfs, "run", evidence)
    shown = SourceCatalog(sources=full.sources, bundles=[
        EvidenceBundle("b", todo.research_question, "", ["S2"], [])])
    issue = {"route": "chroma", "todo_id": "b", "item": "what 403 means",
             "missing_points": ["refuses to authorize"]}

    located = locate_with_index(vfs, index, "run", full, shown, [issue], artifacts, limit=1)

    assert located.bundles[0].source_ids == ["S2", "S1"]
    raw = json.loads(vfs.read("run", "research/b/raw/0.json"))
    assert raw["source"]["locator"] == full.sources["S1"].url


@pytest.mark.parametrize("payload,outcome", [
    ({"approved": True}, "approve"), ({"reply": "approve"}, "approve"), ({"reply": "Đồng ý"}, "approve"),
    ({"approved": False}, "decline"), ({"reply": "cancel"}, "decline"),
    ({"reply": "Only use official documentation"}, "change"),
    ({"approved": False, "reply": "Add a task about caching"}, "change"),
])
def test_plan_decision_outcomes(payload, outcome):
    assert PlanApprovalDecision.model_validate(payload).outcome == outcome


def test_plan_decision_rejects_empty_or_contradictory_input():
    with pytest.raises(ValueError):
        PlanApprovalDecision.model_validate({})
    with pytest.raises(ValueError):
        PlanApprovalDecision.model_validate({"approved": True, "reply": "but change task 2"})


def test_recovery_service_validates_plan_approval_resumes():
    from intelligence_collector.graph.recovery import InvalidRecoveryOperation, WorkflowRecoveryService

    assert WorkflowRecoveryService._validated_resume("plan_approval", {"reply": "approve"})["reply"] == "approve"
    with pytest.raises(ValueError):
        WorkflowRecoveryService._validated_resume("plan_approval", {})
    with pytest.raises(InvalidRecoveryOperation):
        WorkflowRecoveryService._validated_resume("unknown", {"reply": "x"})


def test_url_keyed_registry_entries_survive_credential_redaction(tmp_path):
    """A captured page whose URL contains "oauth"/"token" must keep its registry record;
    observed live: the redacted entry crashed the next research attempt."""
    vfs = VFS(tmp_path)
    entry = {"source_id": "s", "artifact_id": "a", "content_hash": "h", "task_id": "t"}
    vfs.write_json("run", "research/url_registry.json", {
        "https://zuplo.com/learn/mcp/errors/oauth-completes-but-no-token-sent": entry,
        "nested": {"access_token": "secret"}}, "url_registry")

    saved = json.loads(vfs.read("run", "research/url_registry.json"))

    assert saved["https://zuplo.com/learn/mcp/errors/oauth-completes-but-no-token-sent"] == entry
    assert saved["nested"] == {"access_token": "[REDACTED]"}
