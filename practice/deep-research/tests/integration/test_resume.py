from intelligence_collector.storage.checkpoints import Checkpoints
def test_resume_restores_state(tmp_path):
    cp = Checkpoints(tmp_path / "checkpoints.json"); cp.save("thread", {"stage": "Researching", "done": ["t1"]})
    assert cp.load("thread")["done"] == ["t1"]
