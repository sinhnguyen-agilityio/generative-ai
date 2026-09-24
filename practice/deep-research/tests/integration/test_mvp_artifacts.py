from intelligence_collector.agents.scope import build_scope
from intelligence_collector.agents.synthesis import synthesize
from intelligence_collector.contracts import IndexedChunk
def test_stage_artifacts_are_written(stores):
    vfs, _ = stores; run_id = "artifact-run"
    scope = build_scope({k: k for k in ("objective", "research_boundaries", "constraints", "success_criteria", "deliverable_format")}, vfs, run_id)
    chunk = IndexedChunk(source_id="s", artifact_id="a", text="evidence")
    synthesize(scope, [chunk], vfs, run_id)
    assert vfs.read(run_id, "scope/brief_summary.v1.json")
    assert vfs.read(run_id, "reports/draft.v1.md")
