"""The ``write_todos`` tool: persists the planner's todo list."""

from intelligence_collector.contracts import ResearchTodo, TodoStatus
from intelligence_collector.ports import ArtifactStore


def write_todos(vfs: ArtifactStore, run_id: str, todos: list[ResearchTodo]):
    """Validate and save the planned todos to ``planning/todos.v1.json``. Raises ``ValueError`` for
    an empty list, a blank question, or a todo that is not ``planned``.
    """
    if not todos:
        raise ValueError("todo list must not be empty")
    if any(not t.research_question.strip() or t.status != TodoStatus.PLANNED for t in todos):
        raise ValueError("invalid todo")
    return vfs.write_json(
        run_id, "planning/todos.v1.json", [t.model_dump(mode="json") for t in todos], "todo_list"
    )
