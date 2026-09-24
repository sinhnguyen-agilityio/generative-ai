import pytest
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.storage.vector_index import LocalVectorIndex
from tests.fixtures.fake_mcp import FakeMCP
from tests.fixtures.fake_models import FakeModel
from tests.fixtures.fake_vector import FakeVectorIndex

@pytest.fixture
def stores(tmp_path): return VFS(tmp_path / "vfs"), LocalVectorIndex(tmp_path / "index.json")

@pytest.fixture
def fake_mcp(): return FakeMCP()

@pytest.fixture
def fake_model(): return FakeModel()

@pytest.fixture
def fake_vector(): return FakeVectorIndex()
