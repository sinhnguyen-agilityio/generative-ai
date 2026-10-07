"""Checkpoints older than evidence points keep working (spec 002, research R2)."""

from intelligence_collector.contracts import ResearchTodo
from intelligence_collector.graph.migration import normalize_checkpoint


def test_legacy_todo_gets_one_essential_point_linked_to_every_required_answer():
    state = normalize_checkpoint(
        {
            "scoped_request": {
                "research_question": "Compare SQLite and PostgreSQL for mobile apps.",
                "required_answers": ["maximum size", "licence"],
            },
            "plan": {"plan_hash": "abc"},
            "todos": [{"todo_id": "t1", "research_question": "Q?", "evidence_needed": "docs"}],
        }
    )

    todo = ResearchTodo.model_validate(state["todos"][0])
    assert [(p.point_id, p.text, p.required_answers) for p in todo.evidence_points] == [
        ("E1", "docs", ["maximum size", "licence"])
    ]
    assert todo.evidence_points[0].essential
    assert state["plan"] == {"plan_hash": "abc"}  # the approved plan is never rewritten
    assert (state["research_round"], state["gap_stall_count"], state["follow_up_count"]) == (1, 0, 0)
    assert state["gap_analysis"] is None and state["claim_support_score"] is None


def test_brief_without_required_answers_links_to_its_question_and_new_todos_are_untouched():
    point = {"point_id": "E1", "text": "x", "required_answers": []}
    state = normalize_checkpoint(
        {
            "scoped_request": {"research_question": "What is HTTP 403?"},
            "todos": [
                {"todo_id": "t1", "research_question": "What is 403?"},
                {"todo_id": "t2", "research_question": "Other", "evidence_points": [point]},
            ],
        }
    )
    assert state["todos"][0]["evidence_points"][0]["required_answers"] == ["What is HTTP 403?"]
    assert state["todos"][0]["evidence_points"][0]["text"] == "What is 403?"
    assert state["todos"][1]["evidence_points"] == [point]
