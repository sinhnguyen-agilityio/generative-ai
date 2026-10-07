"""Planner Agent: turns the research brief into the research todo list."""

import hashlib
import json

from intelligence_collector.agents.prompts import FEEDBACK_BLOCK, PLANNER_TEMPLATE
from intelligence_collector.contracts import (
    MAX_PLANNED_QUESTIONS,
    ResearchBrief,
    ResearchTodo,
    TaskOrigin,
    TodoPlan,
    evidence_points_from_drafts,
)
from intelligence_collector.ports import ArtifactStore, ModelRouterPort
from intelligence_collector.tools.write_todos import write_todos


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "(none)"


def _numbered(items: list[str]) -> str:
    """Required answers as "1. ..." lines: evidence points refer to them by number."""
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1)) or "(none)"


def plan_record(
    brief: ResearchBrief, todos: list[ResearchTodo], report_shape: str, *, kind: str = "initial"
) -> dict:
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
        # Evidence points are part of the approved strategy: changing one needs a new approval.
        "tasks": [
            {
                "todo_id": todo.todo_id,
                "question": todo.research_question,
                "evidence_needed": todo.evidence_needed,
                "evidence_points": [
                    point.model_dump(mode="json", exclude={"schema_version"})
                    for point in todo.evidence_points
                ],
            }
            for todo in todos
            if todo.origin == TaskOrigin.PLAN
        ],
        "report_shape": report_shape,
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    # Follow-up tasks created by gap analysis stay inside the approved scope, so they are listed
    # for review but excluded from the hash.
    follow_ups = [
        {
            "todo_id": todo.todo_id,
            "question": todo.research_question,
            "round": todo.round,
            "gap_point_ids": list(todo.gap_point_ids),
            "evidence_points": [
                point.model_dump(mode="json", exclude={"schema_version"})
                for point in todo.evidence_points
            ],
        }
        for todo in todos
        if todo.origin == TaskOrigin.FOLLOW_UP
    ]
    return {**body, "plan_hash": digest, "follow_ups": follow_ups}


class PlannerAgent:
    """Generates the research todo list for a brief and saves it to the VFS."""

    def __init__(self, router: ModelRouterPort):
        self.router = router

    def plan(
        self,
        brief: ResearchBrief,
        vfs: ArtifactStore,
        run_id: str,
        *,
        feedback: str = "",
        max_todos: int = MAX_PLANNED_QUESTIONS,
        thread_id: str = "",
    ) -> tuple[list[ResearchTodo], str]:
        """Generate at most `max_todos` research tasks for `brief`, taking the human `feedback`
        into account, and save them through ``write_todos``; returns ``(todos, report_shape)``."""
        if not 1 <= max_todos <= MAX_PLANNED_QUESTIONS:
            raise ValueError(f"max_todos must be between 1 and {MAX_PLANNED_QUESTIONS}")
        messages = PLANNER_TEMPLATE.format_messages(
            max_todos=max_todos,
            brief=brief.research_question,
            required_answers=_numbered(brief.checklist),
            output_requirements=_bullets(brief.output_requirements),
            feedback=FEEDBACK_BLOCK.format(feedback=feedback.strip()) if feedback.strip() else "",
        )
        plan = self.router.invoke_messages(
            "planner", messages, TodoPlan, run_id=run_id, thread_id=thread_id
        )
        if len(plan.questions) > max_todos:
            raise ValueError(f"planner returned {len(plan.questions)} tasks, max is {max_todos}")
        todos = [
            ResearchTodo(
                research_question=question,
                evidence_needed=needed or "; ".join(point.text for point in points),
                evidence_points=points,
            )
            for question, needed, points in zip(
                plan.questions,
                [*plan.evidence_needed, *[""] * len(plan.questions)],
                (evidence_points_from_drafts(d, brief.checklist) for d in plan.evidence_points),
            )
        ]
        # A plan revised from feedback (e.g. a replacement plan) may target only part of the
        # checklist; a fresh plan must serve every required answer.
        linked = {answer for todo in todos for p in todo.evidence_points for answer in p.required_answers}
        unlinked = [answer for answer in brief.checklist if answer not in linked]
        if unlinked and not feedback.strip():
            raise ValueError("required answers without an evidence point: " + "; ".join(unlinked))
        write_todos(vfs, run_id, todos)
        return todos, plan.report_shape
