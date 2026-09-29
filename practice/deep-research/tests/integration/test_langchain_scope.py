import json

import pytest

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from intelligence_collector.agents.scope import ScopeAgent
from intelligence_collector.graph.build import build_graph
from intelligence_collector.models.factory import ModelConfig, ModelFactory
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.storage.vfs import VFS


class RecordingChatModel:
    model_name = "fake-live-scope"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages):
        self.calls.append(list(messages))
        return AIMessage(content=json.dumps(next(self.responses)))


def _guardrail_payload():
    return {"allowed": True, "reason": ""}


def _clear_payload():
    return {
        "need_clarification": False,
        "question": "",
        "verification": (
            "I have enough information to compare PostgreSQL and SQLite for local Python "
            "applications, and I will now begin the research."
        ),
    }


def _brief_payload():
    return {
        "research_question": (
            "I want to compare PostgreSQL and SQLite for local Python applications using current "
            "technical primary sources, with every factual claim cited, and receive an "
            "evidence-backed technical report."
        )
    }


def _plan_payload():
    return {
        "questions": [
            "How do PostgreSQL and SQLite differ in concurrency and locking for local Python "
            "applications, according to their official documentation?",
            "What operational and deployment trade-offs do PostgreSQL and SQLite have for local "
            "Python applications, according to primary technical sources?",
        ],
    }


def test_live_scope_uses_factory_messages_and_selects_planning_edge(monkeypatch, tmp_path):
    calls = []
    model = RecordingChatModel([_guardrail_payload(), _clear_payload(), _brief_payload(), _plan_payload()])
    monkeypatch.setattr(
        "intelligence_collector.models.factory.init_chat_model",
        lambda *args, **kwargs: calls.append((args, kwargs)) or model,
    )
    live_model = ModelFactory().create(ModelConfig("scope-test", provider="fake", timeout=3))
    traces = []
    router = ModelRouter(live_model, trace=lambda **event: traces.append(event))
    result = build_graph(scope_agent=ScopeAgent(router), vfs=VFS(tmp_path)).invoke(
        {"run_id": "run-1", "thread_id": "thread-1", "request": "Compare PostgreSQL and SQLite"}
    )

    assert calls and calls[0][0] == ("scope-test",)
    assert result["stage"] == "Planning"
    question = result["scoped_request"]["research_question"]
    assert question.startswith("I want to compare PostgreSQL")
    assert len(model.calls) == 4
    assert all(isinstance(messages[0], SystemMessage) for messages in model.calls)
    assert any(isinstance(message, HumanMessage) for message in model.calls[0])
    assert question in model.calls[3][-1].content
    assert [trace["role"] for trace in traces[:4]] == ["guardrail", "scope", "scope", "planner"]
    assert traces[0]["selected_model"] == "fake-live-scope"
    assert len(result["messages"]) == 2
    assert isinstance(result["messages"][0], HumanMessage)
    assert isinstance(result["messages"][1], AIMessage)
    assert result["messages"][1].content == _clear_payload()["verification"]
    assert [todo["research_question"] for todo in result["todos"]] == _plan_payload()["questions"]
    assert all(todo["status"] == "planned" for todo in result["todos"])
    assert len({todo["todo_id"] for todo in result["todos"]}) == 2
    assert (tmp_path / "run-1" / "scope" / "brief_summary.v1.json").exists()
    persisted = json.loads((tmp_path / "run-1" / "planning" / "todos.v1.json").read_text())
    assert [todo["research_question"] for todo in persisted] == _plan_payload()["questions"]


def test_invalid_planner_output_does_not_write_todos(tmp_path):
    duplicate = "What are the documented locking differences between PostgreSQL and SQLite?"
    model = RecordingChatModel([
        _guardrail_payload(),
        _clear_payload(),
        _brief_payload(),
        {"questions": [duplicate, duplicate]},
        {"questions": []},
    ])
    graph = build_graph(scope_agent=ScopeAgent(ModelRouter(model)), vfs=VFS(tmp_path))
    with pytest.raises(ValueError, match="invalid structured output for planner"):
        graph.invoke({"run_id": "bad-plan", "thread_id": "bad-plan", "request": "Bounded request"})
    assert not (tmp_path / "bad-plan" / "planning" / "todos.v1.json").exists()


def test_ambiguous_scope_pauses_then_resumes_same_thread_to_planning(tmp_path):
    model = RecordingChatModel([
        _guardrail_payload(),
        {
            "need_clarification": True,
            "question": "Which database versions and workload should be compared?",
            "verification": "",
        },
        _guardrail_payload(),
        {
            "need_clarification": True,
            "question": "What evidence standard should the report meet?",
            "verification": "",
        },
        _guardrail_payload(),
        _clear_payload(),
        _brief_payload(),
        _plan_payload(),
    ])
    router = ModelRouter(model)
    graph = build_graph(scope_agent=ScopeAgent(router), vfs=VFS(tmp_path))
    first = graph.invoke({"run_id": "run-2", "thread_id": "thread-2", "request": "Compare databases"})
    assert first["stage"] == "Scoping"
    assert first["scope_assessment"]["need_clarification"] is True
    assert first["messages"][-1].content.startswith("Which database versions")

    second = graph.invoke({**first, "request": "Compare PostgreSQL and SQLite for local apps"})
    assert second["stage"] == "Scoping"
    assert len(second["messages"]) == 4

    resumed = graph.invoke({**second, "request": "Every factual claim must cite a primary source"})
    assert resumed["stage"] == "Planning"
    assert resumed["scope_assessment"]["need_clarification"] is False
    assert len(resumed["messages"]) == 6
    assert isinstance(resumed["messages"][-2], HumanMessage)
    assert isinstance(resumed["messages"][-1], AIMessage)
    assert resumed["messages"][-1].content == _clear_payload()["verification"]


def test_primary_model_failure_uses_fallback_without_silent_clear(tmp_path):
    class FailingModel:
        model_name = "failed-primary"

        def invoke(self, messages):
            raise TimeoutError("primary timed out")

    fallback = RecordingChatModel([_guardrail_payload(), _clear_payload(), _brief_payload(), _plan_payload()])
    router = ModelRouter(FailingModel(), fallback)
    result = build_graph(scope_agent=ScopeAgent(router), vfs=VFS(tmp_path)).invoke(
        {"run_id": "fallback", "thread_id": "fallback", "request": "A fully bounded request"}
    )
    assert result["stage"] == "Planning"


def test_malformed_structured_output_never_routes_to_planning(tmp_path):
    model = RecordingChatModel([
        _guardrail_payload(),
        {"need_clarification": False},
        {"need_clarification": False},
    ])
    graph = build_graph(scope_agent=ScopeAgent(ModelRouter(model)), vfs=VFS(tmp_path))
    with pytest.raises(ValueError, match="invalid structured output"):
        graph.invoke({"run_id": "bad", "thread_id": "bad", "request": "Research this"})
    assert not (tmp_path / "bad" / "planning" / "todos.v1.json").exists()


def test_total_model_failure_is_visible_and_does_not_write_todos(tmp_path):
    class FailingModel:
        def invoke(self, messages):
            raise RuntimeError("provider unavailable")

    graph = build_graph(
        scope_agent=ScopeAgent(ModelRouter(FailingModel(), FailingModel())),
        vfs=VFS(tmp_path),
    )
    with pytest.raises(ValueError, match="invalid structured output"):
        graph.invoke({"run_id": "failed", "thread_id": "failed", "request": "Research this"})
    assert not (tmp_path / "failed" / "planning" / "todos.v1.json").exists()
