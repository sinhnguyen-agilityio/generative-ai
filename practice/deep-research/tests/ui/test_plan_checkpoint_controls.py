from types import SimpleNamespace

from intelligence_collector import app


def test_pending_plan_node_offers_continuation_instead_of_approval(tmp_path, monkeypatch):
    buttons = []
    st = SimpleNamespace(
        session_state=SimpleNamespace(scope_graph=object(), run_registry=object()),
        button=lambda label: buttons.append(label) or False,
    )
    monkeypatch.setattr(
        app.Settings,
        "from_env",
        lambda: SimpleNamespace(
            checkpoint_path=str(tmp_path / "missing.sqlite"),
            run_registry_path=str(tmp_path / "missing-registry.sqlite"),
        ),
    )
    monkeypatch.setattr(app, "WorkflowRecoveryService", lambda *args: object())
    checkpoint = SimpleNamespace(
        has_interrupt=False, interrupt_kind="", next_nodes=["await_plan_approval"]
    )
    app._render_recovery_controls(
        st, {"stage": "Awaiting Plan Approval", "thread_id": "thread"}, checkpoint
    )
    assert "Approve research plan" not in buttons
    assert "Continue unfinished run" in buttons


def test_actual_plan_interrupt_is_answered_in_chat_without_recovery_buttons():
    buttons = []
    st = SimpleNamespace(button=lambda label: buttons.append(label) or False)
    checkpoint = SimpleNamespace(
        has_interrupt=True, interrupt_kind="plan_approval", next_nodes=["await_plan_approval"]
    )
    app._render_recovery_controls(st, {"stage": "Awaiting Plan Approval"}, checkpoint)
    # The plan is approved or changed by a chat reply; nothing here may resume the run.
    assert buttons == []
