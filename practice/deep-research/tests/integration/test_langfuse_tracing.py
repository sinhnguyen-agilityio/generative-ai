from langchain_core.callbacks import BaseCallbackHandler

from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.graph.build import build_graph
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.observability.tracing import Tracer
from intelligence_collector.storage.vfs import VFS
from tests.contract.test_observability_contract import FakeLangfuseClient, fake_propagation
from tests.integration.test_langchain_scope import (
    RecordingChatModel,
    _brief_payload,
    _clear_payload,
    _guardrail_payload,
    _plan_payload,
)


class CallbackAwareModel(RecordingChatModel):
    def __init__(self, responses):
        super().__init__(responses)
        self.configs = []

    def invoke(self, messages, config=None):
        self.configs.append(config)
        return super().invoke(messages)


def test_scope_execution_has_root_nodes_model_events_callbacks_and_flush(monkeypatch, tmp_path):
    monkeypatch.setattr("langfuse.propagate_attributes", fake_propagation)
    client = FakeLangfuseClient()
    callback = BaseCallbackHandler()
    tracer = Tracer(client=client, callback_handler=callback)
    model = CallbackAwareModel([_guardrail_payload(), _clear_payload(), _brief_payload(), _plan_payload()])
    router = ModelRouter(
        model,
        trace=tracer.model_event,
        callbacks=tracer.callbacks,
    )
    graph = build_graph(
        scope_agent=ScopeAgent(router),
        vfs=VFS(tmp_path),
        tracer=tracer,
    )

    with tracer.execution("run", "thread", input={"request": "bounded request"}) as execution:
        result = graph.invoke(
            {"run_id": "run", "thread_id": "thread", "request": "bounded request"}
        )
        execution.output = {"stage": result["stage"]}

    names = [record["name"] for record in client.records]
    assert names == [
        "research.execution",
        "graph.clarify_with_user",
        "model.validation.guardrail",
        "model.validation.scope",
        "graph.write_research_brief",
        "model.validation.scope",
        "graph.planning",
        "model.validation.planner",
    ]
    assert all(record["parent"] for record in client.records[1:])
    assert len(model.configs) == 4
    assert all(config["callbacks"] == [callback] for config in model.configs)
    assert all(config["metadata"]["langfuse_session_id"] == "thread" for config in model.configs)
    model_records = [record for record in client.records if record["name"] == "model.validation.scope"]
    assert "You are the Scope Agent" in model_records[0]["input"]
    assert "bounded request" in model_records[0]["input"]
    assert '"need_clarification": false' in model_records[0]["updates"][-1]["output"]
    assert "Translate that conversation into one detailed" in model_records[1]["input"]
    assert '"research_question"' in model_records[1]["updates"][-1]["output"]
    planner_record = next(r for r in client.records if r["name"] == "model.validation.planner")
    assert "You are the Planner" in planner_record["input"]
    assert '"questions"' in planner_record["updates"][-1]["output"]
    assert all(isinstance(record["input"], str) for record in client.records)
    assert all(
        isinstance(record["updates"][-1]["output"], str)
        for record in client.records
    )
    assert client.flush_count == 1
