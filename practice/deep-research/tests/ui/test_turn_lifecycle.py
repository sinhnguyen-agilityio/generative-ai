"""A chat turn never leaves an agent shown as running: inputs are disabled while the graph runs,
and a turn that is stopped halfway marks its unfinished nodes as stopped."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from streamlit.runtime.scriptrunner_utils.exceptions import StopException
from streamlit.testing.v1 import AppTest

from intelligence_collector.ui import resources, state, turn
from intelligence_collector.ui.run_status import agent_status
from tests.ui.test_chat_progress import _visible_text
from intelligence_collector.agents.claim_judge import ClaimJudge
from tests.fixtures.fake_models import FakeClaimModel

APP_PATH = Path(__file__).parents[2] / "app.py"


@pytest.fixture(autouse=True)
def disable_local_dotenv(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")


@pytest.fixture
def research_graph(tmp_path, monkeypatch):
    """The real graph with fake models and a controlled search, wired to the UI's lifecycle."""
    from intelligence_collector.agents.report_judge import ReportJudge
    from intelligence_collector.agents.report_writer import ReportWriter
    from intelligence_collector.agents.researcher import ResearchAgent
    from intelligence_collector.agents.scope import ScopeAgent
    from intelligence_collector.graph.build import build_graph
    from intelligence_collector.graph.nodes import ResearchNodes
    from intelligence_collector.models.router import ModelRouter
    from intelligence_collector.observability.tracing import RunTracer
    from intelligence_collector.storage.vector_index import LocalVectorIndex
    from intelligence_collector.storage.vfs import VFS
    from intelligence_collector.tools.mcp import ControlledMCP
    from tests.fixtures.fake_models import FakeAcceptanceModel, FakeReportModel, FakeResearchModel
    from tests.integration.test_full_graph_workflow import (
        REPORT,
        URL,
        SequencedModel,
        _scope_responses,
    )

    monkeypatch.setenv("INTELLIGENCE_VFS_ROOT", str(tmp_path / "vfs"))
    vfs = VFS(tmp_path / "vfs")
    nodes = ResearchNodes(
        vfs,
        LocalVectorIndex(tmp_path / "index.json"),
        ResearchAgent(
            ModelRouter(FakeResearchModel()),
            lambda: ControlledMCP([{"url": URL, "content": "A cache stores reusable data."}]),
        ),
        report_writer=ReportWriter(ModelRouter(FakeReportModel([REPORT]))),
        report_judge=ReportJudge(
            ModelRouter(
                FakeAcceptanceModel(
                    [{"results": [{"criterion": "all", "met": True, "justification": "Ok."}]}]
                )
            )
        ),
        faithfulness_evaluator=lambda q, r, c: {"score": 0.96, "unsupported": []},
        require_faithfulness=True,
        claim_judge=ClaimJudge(ModelRouter(FakeClaimModel())),
    )
    graph = build_graph(
        scope_agent=ScopeAgent(ModelRouter(SequencedModel(_scope_responses()))),
        vfs=vfs,
        research_nodes=nodes,
        checkpointer=MemorySaver(),
        with_approval=True,
        lifecycle=resources._notify_lifecycle,
    )

    def inspect(snapshot):
        current = graph.get_state({"configurable": {"thread_id": snapshot.get("thread_id", "")}})
        kinds = [i.value["kind"] for task in current.tasks for i in task.interrupts]
        return SimpleNamespace(
            interrupt_kind=kinds[0] if kinds else "",
            next_nodes=list(current.next),
            has_interrupt=bool(kinds),
            terminal=False,
        )

    monkeypatch.setattr(resources, "get_graph", lambda: graph)
    monkeypatch.setattr(resources, "inspect_checkpoint", inspect)
    monkeypatch.setattr(resources, "get_tracer", lambda: RunTracer())
    return graph


def test_a_full_run_through_the_ui_leaves_no_agent_running(research_graph):
    app = AppTest.from_file(APP_PATH, default_timeout=60).run()
    app.chat_input[0].set_value("How does a local software cache store and reuse data?").run()
    assert app.session_state["scope_state"]["stage"] == "Planning"

    next(b for b in app.button if b.label == "Approve plan").click().run()

    assert not app.exception and not app.error
    snapshot = app.session_state["scope_state"]
    assert snapshot["stage"] == "Awaiting Approval"
    assert all(item["status"] != "running" for item in app.session_state["activity"])
    assert agent_status(snapshot, "Research Supervisor") == "complete"
    text = _visible_text(app)
    assert 'Researcher</span><span class="ic-chip ic-ok">Complete</span>' in text
    assert "ic-chip ic-run" not in text  # no agent card says Running


def test_a_turn_stopped_halfway_shows_stopped_not_running(monkeypatch):
    session = {
        state.ACTIVITY: [
            {"id": "old:researching", "agent": "Research Supervisor", "status": "success"},
        ],
        state.PROGRESS: "",
    }
    monkeypatch.setattr(turn, "st", SimpleNamespace(session_state=session))

    def interrupted(reply, **kwargs):
        session[turn.EXECUTION_ID] = "now"
        session[state.ACTIVITY].append(
            {"id": "now:researching", "agent": "Research Supervisor", "status": "running"}
        )
        raise StopException()  # what Streamlit raises when the script is stopped

    monkeypatch.setattr(turn, "_run_chat_turn", interrupted)
    with pytest.raises(StopException):
        turn.run_chat_turn(state.Reply("approve", True), view=None, panel=None)

    statuses = [item["status"] for item in session[state.ACTIVITY]]
    assert statuses == ["success", "stopped"]  # only this turn's unfinished node changes
    assert "Continue unfinished run" in session[state.PROGRESS]
    assert agent_status({"activity": session[state.ACTIVITY]}, "Research Supervisor") == "stopped"


def test_inputs_are_disabled_while_a_turn_runs_and_reenabled_after_an_error(monkeypatch):
    checkpoint = SimpleNamespace(
        interrupt_kind="plan_approval",
        next_nodes=["await_plan_approval"],
        has_interrupt=True,
        terminal=False,
    )
    monkeypatch.setattr(resources, "inspect_checkpoint", lambda snapshot: checkpoint)
    seen = []

    def failing_turn(reply, *, view, panel):
        seen.append((reply, view.busy))
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(turn, "run_chat_turn", failing_turn)
    app = AppTest.from_file(APP_PATH).run()
    app.session_state["scope_state"] = {
        "run_id": "run-ui",
        "thread_id": "thread-ui",
        "stage": "Planning",
        "plan": {"tasks": [{"question": "How does caching work?"}]},
    }
    app.session_state["stage"] = "Planning"
    app.run()

    next(b for b in app.button if b.label == "Approve plan").click().run()

    assert seen == [(("approve", True), True)]  # the turn ran in a rerun that rendered everything busy
    assert next(b for b in app.button if b.label == "Approve plan").disabled
    assert next(b for b in app.button if b.label == "New thread").disabled
    assert app.chat_message[-1].markdown[0].value == "approve"  # the reply being answered
    assert "model unavailable" in app.error[0].value
    assert not app.chat_input[0].proto.disabled  # after the error the user can reply again
