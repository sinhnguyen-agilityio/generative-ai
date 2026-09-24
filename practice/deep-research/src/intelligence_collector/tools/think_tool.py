from intelligence_collector.contracts import ProgressAssessment, ResearchTodo, TodoStatus
def think_tool(todos: list[ResearchTodo], artifact_count: int, index_count: int) -> ProgressAssessment:
    complete = [t.todo_id for t in todos if t.status == TodoStatus.COMPLETED]
    ready = bool(todos) and len(complete) == len(todos) and artifact_count > 0 and index_count > 0
    return ProgressAssessment(completed_todos=complete, ready_for_synthesis=ready, next_action="synthesize" if ready else "continue research")

