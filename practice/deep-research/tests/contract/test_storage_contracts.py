import json
import pytest
from intelligence_collector.contracts import IndexedChunk

def test_vfs_writes_json_and_rejects_path_escape(stores):
    vfs, _ = stores
    vfs.write_json("run", "scope/brief.json", {"ok": True}, "brief_summary")
    assert json.loads(vfs.read("run", "scope/brief.json"))["ok"]
    with pytest.raises(ValueError): vfs.write("run", "../escape", "bad", "x")

def test_vector_chunks_keep_provenance(stores):
    _, index = stores
    chunk = IndexedChunk(source_id="source", artifact_id="artifact", text="evidence")
    assert index.index([chunk]) == [chunk.chunk_id]
    assert index.retrieve("evidence")[0].source_id == "source"
