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

    def define_questions(
        self,
        brief: ResearchBrief,
        *,
        run_id: str = "",
        thread_id: str = "",
    ) -> list[str]:
        """Structured LLM decision: decompose the brief into standalone research questions."""
        messages = PLANNER_TEMPLATE.format_messages(brief=brief.research_question)
        plan = self.router.invoke_messages(
            "planner",
            messages,
            TodoPlan,
            prompt_version=PLANNER_PROMPT_VERSION,
            run_id=run_id,
            thread_id=thread_id,
        )
        return plan.questions

    def write_todos(
        self,
        brief: ResearchBrief,
        vfs: VFS,
        run_id: str,
        *,
        thread_id: str = "",
    ) -> list[ResearchTodo]:
        """Define the questions, then durably record them as planned todos."""
        questions = self.define_questions(brief, run_id=run_id, thread_id=thread_id)
        return create_todos(vfs, run_id, questions)


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
