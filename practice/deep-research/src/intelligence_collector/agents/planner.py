"""Planner Agent: decomposes the research brief into the research todo list."""

from intelligence_collector.agents.prompts import PLANNER_TEMPLATE
from intelligence_collector.contracts import ResearchBrief, ResearchTodo, TodoPlan
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.tools.write_todos import write_todos


def create_todos(vfs, run_id, questions: list[str]) -> list[ResearchTodo]:
    """Turn research questions into planned todos and persist them through `write_todos`."""
    todos = [ResearchTodo(research_question=q) for q in questions if q.strip()]
    write_todos(vfs, run_id, todos)
    return todos


class PlannerAgent:
    """Receives the validated research brief and defines the research todo list."""

    def __init__(self, router):
        self.router = router

    def define_plan(
        self,
        brief: ResearchBrief,
        *,
        run_id: str = "",
        thread_id: str = "",
    ) -> TodoPlan:
        """Structured LLM decision: the standalone research questions for the brief."""
        messages = PLANNER_TEMPLATE.format_messages(brief=brief.research_question)
        return self.router.invoke_messages(
            "planner",
            messages,
            TodoPlan,
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
    ) -> list[ResearchTodo]:
        """Define the plan, then durably record the todos."""
        plan = self.define_plan(brief, run_id=run_id, thread_id=thread_id)
        return create_todos(vfs, run_id, plan.questions)

