import pytest
from pydantic import ValidationError

from intelligence_collector.contracts import (
    MAX_ADDITIONAL_RECOVERY_ATTEMPTS,
    MAX_RECOVERY_FEEDBACK_LENGTH,
    RecoveryAction,
    RecoveryDecision,
    WorkflowStatus,
)


def test_workflow_status_distinguishes_waiting_and_terminal_outcomes():
    assert WorkflowStatus.WAITING_FOR_CLARIFICATION.value == "waiting_for_clarification"
    assert WorkflowStatus.WAITING_FOR_APPROVAL.value == "waiting_for_approval"
    assert WorkflowStatus.WAITING_FOR_RECOVERY.value == "waiting_for_recovery"
    assert WorkflowStatus.PUBLISHED != WorkflowStatus.EXPORTED_INCOMPLETE
    assert WorkflowStatus.PUBLISHED != WorkflowStatus.CANCELLED


@pytest.mark.parametrize(
    ("action", "feedback"),
    [
        (RecoveryAction.REVISE, "Use a 100-word executive summary."),
        (RecoveryAction.RESEARCH, "Find primary evidence for the latency claim."),
        (RecoveryAction.RESCOPE, "Compare only the current LTS releases."),
    ],
)
def test_instruction_actions_require_and_normalize_feedback(action, feedback):
    decision = RecoveryDecision(action=action, feedback=f"  {feedback}  ")

    assert decision.feedback == feedback
    assert decision.additional_attempts == 0

    with pytest.raises(ValidationError, match="requires non-empty feedback"):
        RecoveryDecision(action=action, feedback="   ")


def test_retry_report_requires_a_bounded_fresh_allowance():
    decision = RecoveryDecision(
        action=RecoveryAction.RETRY_REPORT,
        additional_attempts=MAX_ADDITIONAL_RECOVERY_ATTEMPTS,
    )
    assert decision.additional_attempts == MAX_ADDITIONAL_RECOVERY_ATTEMPTS

    with pytest.raises(ValidationError, match="at least one additional attempt"):
        RecoveryDecision(action=RecoveryAction.RETRY_REPORT)
    with pytest.raises(ValidationError):
        RecoveryDecision(
            action=RecoveryAction.RETRY_REPORT,
            additional_attempts=MAX_ADDITIONAL_RECOVERY_ATTEMPTS + 1,
        )


@pytest.mark.parametrize(
    "action",
    [
        RecoveryAction.REVISE,
        RecoveryAction.RESEARCH,
        RecoveryAction.RESCOPE,
        RecoveryAction.EXPORT_INCOMPLETE,
        RecoveryAction.CANCEL,
    ],
)
def test_only_retry_report_can_grant_additional_attempts(action):
    kwargs = {"feedback": "Change the requested output."} if action in {
        RecoveryAction.REVISE,
        RecoveryAction.RESEARCH,
        RecoveryAction.RESCOPE,
    } else {}

    with pytest.raises(ValidationError, match="allowed only for retry_report"):
        RecoveryDecision(action=action, additional_attempts=1, **kwargs)


def test_recovery_decision_rejects_unknown_actions_and_graph_destinations():
    with pytest.raises(ValidationError):
        RecoveryDecision(action="jump_to_node")
    with pytest.raises(ValidationError):
        RecoveryDecision.model_validate({
            "action": "cancel",
            "node": "publishing",
        })


def test_recovery_feedback_has_a_size_boundary():
    decision = RecoveryDecision(
        action=RecoveryAction.REVISE,
        feedback="x" * MAX_RECOVERY_FEEDBACK_LENGTH,
    )
    assert len(decision.feedback) == MAX_RECOVERY_FEEDBACK_LENGTH

    with pytest.raises(ValidationError):
        RecoveryDecision(
            action=RecoveryAction.REVISE,
            feedback="x" * (MAX_RECOVERY_FEEDBACK_LENGTH + 1),
        )
