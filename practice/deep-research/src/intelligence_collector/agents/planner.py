"""Planner Agent: decomposes the research brief into the research todo list."""

import hashlib
import json

from intelligence_collector.agents.prompts import PLANNER_TEMPLATE, REPLACEMENT_PLAN_BLOCK
from intelligence_collector.contracts import ResearchBrief, ResearchTodo, TodoPlan
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.tools.write_todos import write_todos


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "(none)"


def create_todos(vfs, run_id, questions: list[str], evidence_needed: list[str] = ()) -> list[ResearchTodo]:
    """Turn research questions into planned todos and persist them through `write_todos`."""
    needed = list(evidence_needed)
    todos = [ResearchTodo(research_question=q, evidence_needed=needed[i] if i < len(needed) else "")
             for i, q in enumerate(questions) if q.strip()]
    write_todos(vfs, run_id, todos)
    return todos


def plan_record(brief: ResearchBrief, todos: list[ResearchTodo], report_shape: str, *,
                kind: str = "initial") -> dict:
    """The plan shown for approval: what is asked, the tasks and the expected report. Its hash
    binds an approval to exactly this plan."""
    body = {
        "kind": kind,
        "original_question": brief.original_question,
        "research_question": brief.research_question,
        "required_answers": brief.checklist,
        "output_requirements": brief.output_requirements,
        # Statuses and repair instructions are left out: reopening an approved task for targeted
        # research stays inside the approved strategy.
        "tasks": [{"todo_id": todo.todo_id, "question": todo.research_question,
                   "evidence_needed": todo.evidence_needed} for todo in todos],
        "report_shape": report_shape,
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return {**body, "plan_hash": digest}


class PlannerAgent:
    """Receives the validated research brief and defines the research todo list."""

    def __init__(self, router):
        self.router = router

    def define_plan(
        self,
        brief: ResearchBrief,
        *,
        unanswered: list[str] = (),
        previous_questions: list[str] = (),
        run_id: str = "",
        thread_id: str = "",
    ) -> TodoPlan:
        """Structured LLM decision: the standalone research questions for the brief. With
        `unanswered`, plan a replacement strategy for those required answers only."""
        replacement = REPLACEMENT_PLAN_BLOCK.format(
            unanswered=_bullets(list(unanswered)), previous=_bullets(list(previous_questions)),
        ) if unanswered else ""
        messages = PLANNER_TEMPLATE.format_messages(
            brief=brief.research_question,
            required_answers=_bullets(brief.checklist),
            output_requirements=_bullets(brief.output_requirements),
            replacement=replacement,
        )
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
        return self.plan(brief, vfs, run_id, thread_id=thread_id)[0]

    def plan(self, brief: ResearchBrief, vfs: VFS, run_id: str, *, thread_id: str = ""):
        """Define and persist the plan; returns ``(todos, report_shape)``."""
        plan = self.define_plan(brief, run_id=run_id, thread_id=thread_id)
        return create_todos(vfs, run_id, plan.questions, plan.evidence_needed), plan.report_shape
