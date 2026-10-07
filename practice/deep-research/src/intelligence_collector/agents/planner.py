"""Planner Agent: decomposes the research brief into the research todo list."""

import hashlib
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from intelligence_collector.ports import ModelRouterPort
from intelligence_collector.agents.prompts import PLANNER_TEMPLATE, REPLACEMENT_PLAN_BLOCK
from intelligence_collector.agents.tools import WRITE_TODOS_TOOL, WRITE_TODOS_TOOL_NAME
from intelligence_collector.contracts import (
    EvidencePointDraft,
    ResearchBrief,
    ResearchTodo,
    TaskOrigin,
    TodoPlan,
    evidence_points_from_drafts,
)
from intelligence_collector.models.redaction import redact_text
from intelligence_collector.models.router import _is_transient
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.tools.write_todos import write_todos


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "(none)"


def _numbered(items: list[str]) -> str:
    """Required answers as "1. ..." lines: evidence points refer to them by number."""
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1)) or "(none)"


def _plan_from(response: AIMessage) -> TodoPlan:
    """Read the plan from the model's ``write_todos`` call. A model that cannot call tools may
    answer with the same arguments as a JSON object instead."""
    calls = [call for call in response.tool_calls if call["name"] == WRITE_TODOS_TOOL_NAME]
    if len(calls) > 1:
        raise ValueError(f"call {WRITE_TODOS_TOOL_NAME} exactly once")
    if calls:
        return TodoPlan.model_validate(calls[0]["args"])
    if response.tool_calls:
        raise ValueError(f"unknown tool {response.tool_calls[0]['name']!r}")
    return TodoPlan.model_validate_json(str(response.content))


def check_links(plan: TodoPlan, checklist: list[str], *, require_all: bool = True) -> None:
    """Raise ``ValueError`` unless every evidence point links only to required answers 1..N
    and (with `require_all`) every required answer is served by at least one point."""
    linked: set[int] = set()
    for drafts in plan.evidence_points:
        for draft in drafts:
            bad = [i for i in draft.required_answers if i > len(checklist)]
            if bad:
                raise ValueError(
                    f"evidence point {draft.text!r} links to {bad}, outside 1..{len(checklist)}"
                )
            linked.update(draft.required_answers)
    unlinked = [item for i, item in enumerate(checklist, start=1) if i not in linked]
    if require_all and unlinked:
        raise ValueError(
            "every required answer needs at least one evidence point; unlinked: "
            + "; ".join(unlinked)
        )


def create_todos(
    vfs,
    run_id,
    questions: list[str],
    evidence_needed: list[str] = (),
    evidence_points: list[list[EvidencePointDraft]] = (),
    checklist: list[str] = (),
) -> list[ResearchTodo]:
    """Turn research questions and their evidence points into planned todos and persist them
    through `write_todos`. Evidence point indexes resolve to `checklist` texts."""
    needed = list(evidence_needed)
    drafts = list(evidence_points)
    todos = []
    for i, question in enumerate(questions):
        if not question.strip():
            continue
        points = evidence_points_from_drafts(drafts[i], list(checklist)) if i < len(drafts) else []
        todos.append(
            ResearchTodo(
                research_question=question,
                evidence_needed=(needed[i] if i < len(needed) and needed[i] else "")
                or "; ".join(point.text for point in points),
                evidence_points=points,
            )
        )
    write_todos(vfs, run_id, todos)
    return todos


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
    """Receives the validated research brief and defines the research todo list."""

    def __init__(self, router: ModelRouterPort):
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
        """Tool-calling LLM decision: the planner calls ``write_todos`` with the standalone
        research questions for the brief. With `unanswered`, plan a replacement strategy for
        those required answers only. An invalid call is answered with its validation error and
        retried, up to the router's ``max_attempts``."""
        replacement = (
            REPLACEMENT_PLAN_BLOCK.format(
                unanswered=_bullets(list(unanswered)),
                previous=_bullets(list(previous_questions)),
            )
            if unanswered
            else ""
        )
        messages = PLANNER_TEMPLATE.format_messages(
            brief=brief.research_question,
            required_answers=_numbered(brief.checklist),
            output_requirements=_bullets(brief.output_requirements),
            replacement=replacement,
        )
        errors = []
        for attempt in range(1, self.router.max_attempts + 1):
            try:
                response = self.router.invoke_with_tools(
                    "planner",
                    messages,
                    [WRITE_TODOS_TOOL],
                    run_id=run_id,
                    thread_id=thread_id,
                    tool_choice=WRITE_TODOS_TOOL_NAME,
                    attempt=attempt,
                )
            except Exception as exc:
                if _is_transient(exc):
                    raise RuntimeError(
                        f"model call for planner failed: {redact_text(exc)}"
                    ) from exc
                errors.append(redact_text(exc))
                continue
            try:
                plan = _plan_from(response)
                # A replacement plan covers only the unanswered items, not the whole checklist.
                check_links(plan, brief.checklist, require_all=not unanswered)
                return plan
            except (ValidationError, ValueError) as exc:
                errors.append(redact_text(exc))
                feedback = f"Invalid {WRITE_TODOS_TOOL_NAME} call: {exc}. Call it again, fixed."
                messages = [
                    *messages,
                    response,
                    *(
                        [
                            ToolMessage(content=feedback, tool_call_id=call["id"])
                            for call in response.tool_calls
                        ]
                        or [HumanMessage(content=feedback)]
                    ),
                ]
        raise ValueError(
            f"invalid structured output for planner after {self.router.max_attempts} attempts: "
            + "; ".join(errors)
        )

    def plan(
        self, brief: ResearchBrief, vfs: VFS, run_id: str, *, thread_id: str = ""
    ) -> tuple[list[ResearchTodo], str]:
        """Define the plan, then durably record its todos through ``write_todos``; returns
        ``(todos, report_shape)``."""
        plan = self.define_plan(brief, run_id=run_id, thread_id=thread_id)
        todos = create_todos(
            vfs, run_id, plan.questions, plan.evidence_needed, plan.evidence_points, brief.checklist
        )
        return todos, plan.report_shape
