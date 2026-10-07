import pytest
from pydantic import ValidationError

from intelligence_collector.contracts import TodoPlan


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
        TodoPlan.model_validate(
            {
                "questions": ["Q?"],
                "todos": [{"research_question": "Q?"}],
            }
        )
    with pytest.raises(ValidationError):
        TodoPlan(questions=["Valid question?"], acceptance_criteria=["Not planned any more."])


def test_todo_plan_output_is_a_list_of_trimmed_questions():
    point = [{"text": "evidence", "required_answers": [1]}]
    plan = TodoPlan(
        questions=["  First question?  ", "Second question?"], evidence_points=[point, point]
    )
    assert plan.questions == ["First question?", "Second question?"]
