"""End-to-end regression test for the graph.invoke() -> graph.stream() switch.

The pipeline rows are redrawn on every step of `graph.stream(state, config,
stream_mode="values")` (see app.py::render), not only once after the whole run finishes. This
test drives a real chat turn through the actual Streamlit script and checks the settled result
still matches what `graph.invoke()` used to produce -- AppTest can only observe the state after
`.run()` returns, so it proves the switch didn't change the outcome; the live, node-by-node
update behavior itself is verified directly against LangGraph's `stream_mode="values"` API
(every yielded step carries the full accumulated state, confirmed while building this feature).
"""

from pathlib import Path

import pytest

from intelligence_collector.observability.tracing import RunTracer
from tests.integration.test_langchain_scope import (
    RecordingChatModel,
    _brief_payload,
    _clear_payload,
    _guardrail_payload,
    _plan_payload,
)

APP_PATH = Path(__file__).parents[2] / "app.py"


@pytest.fixture(autouse=True)
def disable_local_dotenv(monkeypatch):
    """This test must never load a developer's real local .env (TAVILY_API_KEY etc. would
    silently turn on the full research pipeline instead of the scope-only path being tested)."""
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")


def test_chat_turn_streams_through_the_graph_and_settles_on_the_planned_stage(
    tmp_path, monkeypatch
):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("INTELLIGENCE_REASONING_MODEL", "fake-scope")
    monkeypatch.setenv("INTELLIGENCE_REASONING_PROVIDER", "fake")
    monkeypatch.setenv("INTELLIGENCE_STANDARD_MODEL", "fake-standard")
    monkeypatch.setenv("INTELLIGENCE_STANDARD_PROVIDER", "fake")
    monkeypatch.setenv("INTELLIGENCE_VFS_ROOT", str(tmp_path / "vfs"))
    monkeypatch.setenv("INTELLIGENCE_CHECKPOINT_PATH", str(tmp_path / "checkpoints.sqlite3"))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_MCP_URL", raising=False)

    # One shared fake model backs both tiers (init_chat_model is patched to ignore its args),
    # so every invoke_messages call — guardrail, scope, brief, planner — draws from one queue.
    model = RecordingChatModel(
        [_guardrail_payload(), _clear_payload(), _brief_payload(), _plan_payload()]
    )
    monkeypatch.setattr(
        "intelligence_collector.models.factory.init_chat_model",
        lambda *args, **kwargs: model,
    )
    monkeypatch.setattr(
        "intelligence_collector.app._get_tracer",
        lambda st: RunTracer(),
    )

    app = AppTest.from_file(APP_PATH).run()
    app.chat_input[0].set_value("Compare PostgreSQL and SQLite for local apps").run()

    assert not app.exception
    assert not app.error
    assert app.session_state["scope_state"]["stage"] == "Planning"
    assert (
        app.session_state["scope_state"]["scoped_request"]["research_question"]
        == (_brief_payload()["research_question"])
    )
    assert [todo["research_question"] for todo in app.session_state["scope_state"]["todos"]] == (
        _plan_payload()["questions"]
    )
    # The settled pipeline rows reflect the final, fully-streamed state.
    markdown_text = "\n".join(item.value for item in app.markdown)
    assert markdown_text.index("Scope Agent") < markdown_text.index("Planner")
    assert _plan_payload()["questions"][0] in markdown_text
