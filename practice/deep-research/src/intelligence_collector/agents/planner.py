from intelligence_collector.contracts import ResearchTodo
from intelligence_collector.tools.write_todos import write_todos
def plan(vfs, run_id, questions):
    todos = [ResearchTodo(research_question=q) for q in questions if q.strip()]
    write_todos(vfs, run_id, todos)
    return todos

def replan(vfs, run_id, questions):
    """Replace only the affected plan while retaining existing run artifacts."""
    return plan(vfs, run_id, questions)
