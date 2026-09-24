import json
import pytest
from intelligence_collector.contracts import IndexedChunk
from intelligence_collector.storage.artifacts import ArtifactStore
from intelligence_collector.storage.checkpoints import Checkpoints

def test_artifact_store_versions_and_vfs_escape(stores):
    vfs, _ = stores; store = ArtifactStore(vfs)
    first = store.write("run", "brief_summary", "scope/brief.json", {"ok": True})
    assert first.version == 1 and json.loads(vfs.read("run", "scope/brief.json"))["ok"]
    with pytest.raises(ValueError): vfs.write("run", "../escape", "bad", "x")

def test_checkpoint_corruption_is_safe(tmp_path):
    path = tmp_path / "cp.json"; path.write_text("{")
    assert Checkpoints(path).load("missing") is None

def test_vector_chunks_keep_provenance(stores):
    _, index = stores
    chunk = IndexedChunk(source_id="source", artifact_id="artifact", text="evidence")
    assert index.index([chunk]) == [chunk.chunk_id]
    assert index.retrieve("evidence")[0].source_id == "source"
