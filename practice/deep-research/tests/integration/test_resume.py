from langchain_core.messages import AIMessage, HumanMessage

from intelligence_collector.graph.build import build_graph
from intelligence_collector.storage.checkpoints import Checkpoints


def test_resume_restores_state(tmp_path):
    cp = Checkpoints(tmp_path / "checkpoints.json")
    cp.save("thread", {"stage": "Researching", "done": ["t1"]})
    assert cp.load("thread")["done"] == ["t1"]


def test_checkpoint_round_trip_preserves_complete_workflow_state(tmp_path):
    cp = Checkpoints(tmp_path / "checkpoints.json")
    state = {
        "run_id": "run",
        "thread_id": "thread",
        "stage": "Verifying",
        "messages": [HumanMessage(content="request"), AIMessage(content="verification")],
        "scoped_request": {"research_question": "A sufficiently detailed research question"},
        "todos": [{"todo_id": "todo-1", "status": "completed"}],
        "todo_ids": ["todo-1"],
        "artifacts": [{"artifact_id": "artifact-1"}],
        "evidence": [{"artifact_id": "evidence-1"}],
        "index_refs": ["chunk-1"],
        "progress_assessment": {"ready_for_synthesis": True},
        "draft": "latest report",
        "previous_draft": "previous report",
        "verification": {"passed": False},
        "verification_history": [{"passed": False, "artifact_hash": "old"}],
        "recovery_attempt": 2,
        "recovery_budget": 3,
        "progress_fingerprint": "fingerprint",
        "approval": {"approved": False},
        "publication": {"published": False},
    }

    cp.save("thread", state)
    restored = cp.load("thread")

    assert restored["scoped_request"] == state["scoped_request"]
    assert restored["todos"] == state["todos"]
    assert restored["artifacts"] == state["artifacts"]
    assert restored["progress_assessment"] == state["progress_assessment"]
    assert restored["draft"] == "latest report"
    assert restored["verification_history"] == state["verification_history"]
    assert restored["recovery_attempt"] == 2
    assert restored["publication"] == {"published": False}
    assert [message.content for message in restored["messages"]] == [
        "request",
        "verification",
    ]


def test_every_graph_node_loads_and_saves_checkpoint_boundary(tmp_path):
    class RecordingCheckpoints(Checkpoints):
        def __init__(self, path):
            super().__init__(path)
            self.saved_stages = []
            self.load_count = 0

        def load(self, thread_id):
            self.load_count += 1
            return super().load(thread_id)

        def save(self, thread_id, state):
            self.saved_stages.append(state["stage"])
            super().save(thread_id, state)

    checkpoints = RecordingCheckpoints(tmp_path / "node-checkpoints.json")
    result = build_graph(checkpoint_store=checkpoints).invoke({
        "run_id": "run",
        "thread_id": "thread",
        "artifacts": [{"artifact_id": "retained"}],
        "recovery_attempt": 1,
    })

    assert result["stage"] == "verifying"
    assert checkpoints.saved_stages == [
        "scope",
        "planning",
        "researching",
        "synthesizing",
        "verifying",
    ]
    assert checkpoints.load_count == 5
    persisted = checkpoints.load("thread")
    assert persisted["artifacts"] == [{"artifact_id": "retained"}]
    assert persisted["recovery_attempt"] == 1
    assert "verification_history" in persisted
    assert "publication" in persisted
