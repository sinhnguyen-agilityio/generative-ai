"""A Streamlit rerun is a re-render, not a research execution: only a submitted chat message
opens an execution observation in the run's trace."""

from pathlib import Path

import pytest

from tests.fixtures.langfuse_capture import LangfuseCapture
from tests.integration.test_langchain_scope import (
    RecordingChatModel,
    _brief_payload,
    _clear_payload,
    _guardrail_payload,
    _plan_payload,
)

APP_PATH = Path(__file__).parents[2] / "app.py"


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("INTELLIGENCE_REASONING_MODEL", "fake-scope")
    monkeypatch.setenv("INTELLIGENCE_REASONING_PROVIDER", "fake")
    monkeypatch.setenv("INTELLIGENCE_STANDARD_MODEL", "fake-standard")
    monkeypatch.setenv("INTELLIGENCE_STANDARD_PROVIDER", "fake")
    monkeypatch.setenv("INTELLIGENCE_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.setenv("INTELLIGENCE_CHECKPOINT_PATH", str(tmp_path / "checkpoints.sqlite3"))
    monkeypatch.setenv("INTELLIGENCE_RUN_REGISTRY_PATH", str(tmp_path / "runs.sqlite3"))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_MCP_URL", raising=False)


def test_reruns_do_not_start_executions_and_a_message_starts_exactly_one(monkeypatch):
    from streamlit.testing.v1 import AppTest

    capture = LangfuseCapture()
    model = RecordingChatModel(
        [_guardrail_payload(), _clear_payload(), _brief_payload(), _plan_payload()]
    )
    monkeypatch.setattr(
        "intelligence_collector.models.factory.init_chat_model", lambda *args, **kwargs: model
    )
    monkeypatch.setattr("intelligence_collector.ui.resources.get_tracer", lambda: capture.tracer)

    app = AppTest.from_file(APP_PATH).run()
    app.run()  # a refresh
    assert capture.spans() == []

    app.chat_input[0].set_value("Compare PostgreSQL and SQLite for local apps").run()
    assert not app.exception
    executions = capture.executions()
    assert len(executions) == 1 and executions[0].metadata["execution_kind"] == "start"
    run_id = app.session_state["scope_state"]["run_id"]
    assert executions[0].trace_id == capture.tracer.trace_id_for(run_id)
    assert app.session_state["scope_state"]["trace_id"] == executions[0].trace_id
    span_count = len(capture.spans())

    app.run()
    app.run()
    assert len(capture.executions()) == 1
    assert len(capture.spans()) == span_count
    assert len(model.calls) == 4  # no model call repeated by a rerun
