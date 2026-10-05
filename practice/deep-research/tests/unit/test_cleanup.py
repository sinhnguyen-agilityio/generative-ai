import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("filename", ["check_model_fallback.py", "check_faithfulness.py"])
def test_importing_manual_scripts_does_not_make_paid_calls(monkeypatch, filename):
    from intelligence_collector.models.factory import ModelFactory

    def unexpected_call(*args, **kwargs):
        pytest.fail("Importing a manual script must not construct a live model or evaluator")

    monkeypatch.setattr(ModelFactory, "create", unexpected_call)
    monkeypatch.setattr(
        "intelligence_collector.observability.evaluation.make_faithfulness_evaluator",
        unexpected_call,
    )
    path = Path(__file__).resolve().parents[2] / "scripts" / filename
    spec = importlib.util.spec_from_file_location("manual_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert callable(module.main)


def test_recovery_controls_reuse_inspected_checkpoint(monkeypatch):
    from intelligence_collector.ui import resources
    from intelligence_collector.ui.components import recovery

    class Session(dict):
        def __getattr__(self, key):
            return self[key]

    class UI:
        session_state = Session(scope_graph=object(), run_registry=object())
        buttons = []

        def button(self, label, **kwargs):
            self.buttons.append(label)
            return False

    checkpoint = SimpleNamespace(next_nodes=["synthesizing"], has_interrupt=False, terminal=False)
    calls = []

    class Service:
        def __init__(self, *args):
            pass

        def inspect(self, thread):
            calls.append(thread)
            return checkpoint

    ui = UI()
    monkeypatch.setattr(resources, "st", ui)
    monkeypatch.setattr(recovery, "st", ui)
    monkeypatch.setattr(resources, "WorkflowRecoveryService", Service)
    snapshot = {"run_id": "run", "thread_id": "thread", "stage": "Synthesizing"}
    inspected = resources.inspect_checkpoint(snapshot)
    recovery.render_recovery_controls(snapshot, inspected)
    assert calls == ["thread"]
    assert "Continue unfinished run" in ui.buttons
