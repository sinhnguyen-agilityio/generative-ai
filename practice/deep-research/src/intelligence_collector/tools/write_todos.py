from intelligence_collector.contracts import ResearchTodo, TodoStatus
from intelligence_collector.storage.vfs import VFS


def write_todos(vfs: VFS, run_id: str, todos: list[ResearchTodo]):
    if not todos:
        raise ValueError("todo list must not be empty")
    if any(not t.research_question.strip() or t.status != TodoStatus.PLANNED for t in todos):
        raise ValueError("invalid todo")
    return vfs.write_json(run_id, "planning/todos.v1.json", [t.model_dump(mode="json") for t in todos], "todo_list")
