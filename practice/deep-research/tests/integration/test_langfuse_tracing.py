"""Scope and planning under a traced execution: every model call is one Langfuse generation,
with model name, usage and the rendered prompt, nested under the graph node that made it."""

from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.graph.build import build_graph
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.storage.vfs import VFS
from tests.fixtures.chat_model import USAGE, FakeChatModel
from tests.fixtures.langfuse_capture import LangfuseCapture
from tests.integration.test_langchain_scope import (
    RecordingChatModel,
    _brief_payload,
    _clear_payload,
    _guardrail_payload,
    _plan_payload,
)


def test_scope_and_planning_generations_nest_under_their_nodes(tmp_path):
    capture = LangfuseCapture()
    model = FakeChatModel(model_name="fake-scope-model", delegate=RecordingChatModel(
        [_guardrail_payload(), _clear_payload(), _brief_payload(), _plan_payload()]))
    graph = build_graph(scope_agent=ScopeAgent(ModelRouter(model)), vfs=VFS(tmp_path),
                        tracer=capture.tracer)

    with capture.tracer.execution(run_id="run", thread_id="thread", kind="start",
                                  input={"request": "bounded request"}) as execution:
        result = graph.invoke({"run_id": "run", "thread_id": "thread",
                               "request": "bounded request api_key=do-not-export"},
                              execution.graph_config({}))
        execution.finish(result)

    assert result["stage"] == "Planning"
    generations = [span for span in capture.spans() if span.type == "generation"]
    assert [capture.ancestors(g)[:2] for g in generations] == [
        ["model.guardrail", "clarify_with_user"],
        ["model.scope", "clarify_with_user"],
        ["model.scope", "write_research_brief"],
        ["model.planner", "planning"],
    ]
    assert model.calls == 4
    exported = capture.exporter.get_finished_spans()
    generation = next(s for s in exported if s.attributes.get("langfuse.observation.type")
                      == "generation")
    assert generation.attributes["langfuse.observation.model.name"] == "fake-scope-model"
    assert '"input": 12' in generation.attributes["langfuse.observation.usage_details"]
    assert USAGE["output_tokens"] == 5
    planner_input = str(generations[-1].input)
    assert "You are the Planner" in planner_input
    everything = repr([dict(s.attributes or {}) for s in exported])
    assert "do-not-export" not in everything
