import pytest
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.storage.vector_index import LocalVectorIndex
@pytest.fixture
def stores(tmp_path): return VFS(tmp_path / "vfs"), LocalVectorIndex(tmp_path / "index.json")

