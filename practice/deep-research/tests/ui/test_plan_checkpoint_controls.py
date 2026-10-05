from types import SimpleNamespace

from intelligence_collector.ui.components import recovery


def test_pending_plan_node_offers_continuation_instead_of_approval(tmp_path, monkeypatch):
    buttons = []
    monkeypatch.setattr(
        recovery, "st", SimpleNamespace(button=lambda label, **kw: buttons.append(label) or False)
    )
    checkpoint = SimpleNamespace(
        has_interrupt=False, interrupt_kind="", next_nodes=["await_plan_approval"]
    )
    recovery.render_recovery_controls(
        {"stage": "Awaiting Plan Approval", "thread_id": "thread"}, checkpoint
    )
    assert "Approve research plan" not in buttons
    assert "Continue unfinished run" in buttons


def test_actual_plan_interrupt_is_answered_in_chat_without_recovery_buttons(monkeypatch):
    buttons = []
    monkeypatch.setattr(
        recovery, "st", SimpleNamespace(button=lambda label, **kw: buttons.append(label) or False)
    )
    checkpoint = SimpleNamespace(
        has_interrupt=True, interrupt_kind="plan_approval", next_nodes=["await_plan_approval"]
    )
    recovery.render_recovery_controls({"stage": "Awaiting Plan Approval"}, checkpoint)
    # The plan is approved or changed by a chat reply; nothing here may resume the run.
    assert buttons == []
