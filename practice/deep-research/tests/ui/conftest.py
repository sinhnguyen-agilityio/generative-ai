import pytest


@pytest.fixture(autouse=True)
def isolated_workflow_storage(tmp_path, monkeypatch):
    """UI tests must not discover the developer's real unfinished runs."""
    monkeypatch.setenv("INTELLIGENCE_RUN_REGISTRY_PATH", str(tmp_path / "registry.sqlite"))
    monkeypatch.setenv("INTELLIGENCE_CHECKPOINT_PATH", str(tmp_path / "checkpoints.sqlite"))
