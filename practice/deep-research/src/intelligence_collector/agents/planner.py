from intelligence_collector.agents.prompts import PLANNER_PROMPT_VERSION, PLANNER_TEMPLATE
from intelligence_collector.contracts import (
    ResearchBrief,
    ResearchGraphState,
    ResearchTodo,
    Stage,
    TodoPlan,
)
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.tools.write_todos import write_todos


def create_todos(vfs, run_id, questions: list[str]) -> list[ResearchTodo]:
    """Turn research questions into planned todos and persist them through `write_todos`."""
    todos = [ResearchTodo(research_question=q) for q in questions if q.strip()]
    write_todos(vfs, run_id, todos)
    return todos


def replan(vfs, run_id, questions):
    """Replace only the affected plan while retaining existing run artifacts."""
    return create_todos(vfs, run_id, questions)


class PlannerAgent:
    """Receives the validated research brief and defines the research todo list."""

    def __init__(self, router):
        self.router = router

    def define_plan(
        self,
        brief: ResearchBrief,
        *,
        guidance: str = "",
        run_id: str = "",
        thread_id: str = "",
    ) -> TodoPlan:
        """Structured LLM decision: standalone research questions plus acceptance criteria.

        ``guidance``, when non-empty, is a human reviewer's replanning feedback on a previous
        report built from an earlier plan — see ``FEEDBACK_ROUTER_PROMPT``/``graph/nodes.py``'s
        ``ApprovalNodes``. Empty by default (the normal, first-time planning call).
        """
        messages = PLANNER_TEMPLATE.format_messages(
            brief=brief.research_question,
            guidance=(
                f"Reviewer replanning guidance (a previous report was rejected because):\n{guidance}"
                if guidance else ""
            ),
        )
        return self.router.invoke_messages(
            "planner",
            messages,
            TodoPlan,
            prompt_version=PLANNER_PROMPT_VERSION,
            run_id=run_id,
            thread_id=thread_id,
        )

    def write_todos(
        self,
        brief: ResearchBrief,
        vfs: VFS,
        run_id: str,
        *,
        thread_id: str = "",
        guidance: str = "",
    ) -> tuple[list[ResearchTodo], list[str]]:
        """Define the plan, then durably record the todos and acceptance criteria."""
        plan = self.define_plan(brief, guidance=guidance, run_id=run_id, thread_id=thread_id)
        todos = create_todos(vfs, run_id, plan.questions)
        vfs.write_json(
            run_id,
            "planning/acceptance_criteria.v1.json",
            plan.acceptance_criteria,
            "acceptance_criteria",
        )
        return todos, plan.acceptance_criteria


def replan_affected_node(
    state: ResearchGraphState,
    vfs,
    affected_todo_ids: list[str],
    replacement_questions: list[str],
) -> ResearchGraphState:
    """Replace affected tasks without discarding completed, unaffected work."""
    affected = set(affected_todo_ids)
    if not affected:
        raise ValueError("affected-task re-planning requires at least one todo id")
    known = {todo.todo_id for todo in state.todos}
    unknown = affected - known
    if unknown:
        raise ValueError(f"cannot re-plan unknown todo ids: {sorted(unknown)}")
    questions = [question.strip() for question in replacement_questions if question.strip()]
    if not questions:
        raise ValueError("affected-task re-planning requires replacement questions")

    retained = [todo for todo in state.todos if todo.todo_id not in affected]
    replacements = [ResearchTodo(research_question=question) for question in questions]
    version = state.recovery_attempt + 1
    ref = vfs.write_json(
        state.run_id,
        f"planning/replan.v{version}.json",
        {
            "affected_todo_ids": sorted(affected),
            "retained_todo_ids": [todo.todo_id for todo in retained],
            "replacement_todos": [todo.model_dump(mode="json") for todo in replacements],
        },
        "replan",
    )
    todos = [*retained, *replacements]
    return state.model_copy(update={
        "stage": Stage.RESEARCHING,
        "status": "active",
        "todo_ids": [todo.todo_id for todo in todos],
        "todos": todos,
        "artifacts": [*state.artifacts, ref],
        "progress_assessment": None,
        "verification": None,
        "approval": None,
        "publication": None,
        "last_error": "",
    })
