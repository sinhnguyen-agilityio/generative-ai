from intelligence_collector.contracts import ProgressAssessment, ResearchTodo, TodoStatus


def think_tool(todos: list[ResearchTodo], artifact_count: int, index_count: int) -> ProgressAssessment:
    complete = [t.todo_id for t in todos if t.status == TodoStatus.COMPLETED]
    blockers = [
        f"Task not completed ({t.status.value}): {t.research_question}"
        for t in todos
        if t.status != TodoStatus.COMPLETED
    ]
    if not todos:
        blockers.append("No planned research tasks")
    if artifact_count <= 0:
        blockers.append("No research artifacts persisted")
    if index_count <= 0:
        blockers.append("No research data indexed")
    ready = not blockers
    return ProgressAssessment(
        completed_todos=complete,
        blockers=blockers,
        ready_for_synthesis=ready,
        next_action="synthesize" if ready else "continue research",
    )
