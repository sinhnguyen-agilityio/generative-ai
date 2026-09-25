import pytest
from pydantic import ValidationError

from intelligence_collector.contracts import (
    FixDecision,
    SupervisorAssessment,
    TodoPlan,
    VerificationAssessment,
    WorkerSearchPlan,
)


def test_semantic_decisions_reject_workflow_changing_malformed_output():
    with pytest.raises(ValidationError):
        TodoPlan(questions=[])
    with pytest.raises(ValidationError):
        TodoPlan(questions=["Valid question?", "   "])
    with pytest.raises(ValidationError):
        TodoPlan(questions=["Same question?", "same question?"])
    with pytest.raises(ValidationError):
        TodoPlan(questions=[f"Question {n}?" for n in range(6)])
    with pytest.raises(ValidationError):
        TodoPlan.model_validate({"questions": ["Q?"], "todos": [{"research_question": "Q?"}]})


def test_todo_plan_output_is_a_list_of_trimmed_questions():
    plan = TodoPlan(questions=["  First question?  ", "Second question?"])
    assert plan.questions == ["First question?", "Second question?"]
    with pytest.raises(ValidationError):
        WorkerSearchPlan(queries=[], stop_condition="")
    with pytest.raises(ValidationError):
        VerificationAssessment(passed=False, feedback="missing evidence")
    with pytest.raises(ValidationError):
        VerificationAssessment(passed=True, issue_type="DRAFT_DEFECT")
    with pytest.raises(ValidationError):
        SupervisorAssessment(
            coverage_complete=False,
            research_complete=True,
            blockers=["missing task"],
        )
    with pytest.raises(ValidationError):
        FixDecision(action="publish", reason="bypass verification")
