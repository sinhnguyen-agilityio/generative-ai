"""Gap analysis at the end of a research round (spec 002, FR-016 to FR-021).

1. Collect every missing or partial evidence point from the latest report of each active task.
2. Classify it without a model: essential when the approved plan links it to a required answer;
   non-essential gaps become limitations.
3. Judge every essential gap against all tasks' reports: a follow-up task that targeted it and
   reports its points answered (with quotes already verified) covers it. Look the remaining ones
   up in the stored pages of the active tasks; one ``supervisor`` call judges whether a passage
   covers it, and a gap counts as covered only with a verified quote.
4. For gaps still open, the supervisor writes follow-up tasks (``CreateFollowUp``) in at most
   ``MAX_TURNS`` turns; code refuses unknown or covered gaps, links outside the brief, duplicates
   and anything over the caps. A gap outside the approved scope goes to ``ProposeReplan``.
5. The stall counter counts follow-up rounds only (round 1 never counts).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from intelligence_collector.agents.prompts import GAP_COVERAGE_TEMPLATE, GAP_FOLLOW_UP_TEMPLATE
from intelligence_collector.agents.searcher_report import lookup_points, phrasings_for, task_points
from intelligence_collector.agents.tools import (
    CREATE_FOLLOW_UP_TOOL,
    CREATE_FOLLOW_UP_TOOL_NAME,
    PROPOSE_REPLAN_TOOL,
    PROPOSE_REPLAN_TOOL_NAME,
)
from intelligence_collector.contracts import (
    EvidencePointDraft,
    Gap,
    GapAnalysis,
    GapCoverageBatch,
    PassageLookupRecord,
    PointStatus,
    ResearchTodo,
    SearcherReport,
    TaskOrigin,
    evidence_points_from_drafts,
)
from intelligence_collector.models.redaction import redact_error
from intelligence_collector.observability.tracing import traced
from intelligence_collector.ports import ModelRouterPort
from intelligence_collector.source_verification import quote_in_text
from intelligence_collector.storage.passage_lookup import PassageLookup

MAX_TURNS = 4
STALL_LIMIT = 2


def _normalized(text: str) -> str:
    return " ".join(sorted(re.findall(r"\w+", text.casefold())))


@dataclass
class GapResult:
    """One round's analysis, the follow-up tasks it created and any replan proposal."""

    analysis: GapAnalysis
    follow_ups: list[ResearchTodo] = field(default_factory=list)
    replan: dict | None = None
    lookups: list[PassageLookupRecord] = field(default_factory=list)


def collect_gaps(todos: list[ResearchTodo], reports: dict[str, SearcherReport]) -> list[Gap]:
    """Every missing or partial point of the tasks, classified from the plan links (FR-016a)."""
    gaps = []
    for todo in todos:
        report = reports.get(todo.todo_id)
        statuses = {p.point_id: p.status for p in report.points} if report else {}
        for point in task_points(todo):
            status = statuses.get(point.point_id, PointStatus.MISSING)
            if status == PointStatus.ANSWERED:
                continue
            gaps.append(
                Gap(
                    point_key=f"{todo.todo_id}:{point.point_id}",
                    text=point.text,
                    status=status,
                    essential=point.essential,
                    required_answers=list(point.required_answers),
                    resolution="open" if point.essential else "limitation",
                )
            )
    return gaps


class GapAnalyzer:
    """Runs one round's gap analysis for the research supervisor."""

    def __init__(
        self,
        router: ModelRouterPort | None,
        lookup: PassageLookup | None,
        *,
        max_per_answer: int = 2,
        max_per_run: int = 4,
    ):
        self.router = router
        self.lookup = lookup
        self.max_per_answer = max_per_answer
        self.max_per_run = max_per_run

    def analyze(
        self,
        *,
        run_id: str,
        todos: list[ResearchTodo],
        reports: dict[str, SearcherReport],
        checklist: list[str],
        round_number: int,
        previous: dict | None,
        stall_count: int,
        allowed_source_ids: set[str],
        stored_text: Callable[[str], str],
        thread_id: str = "",
    ) -> GapResult:
        gaps = collect_gaps(todos, reports)
        analysis = GapAnalysis(round=round_number, gaps=gaps)
        result = GapResult(analysis=analysis)
        with traced(
            "gap_analysis",
            metadata={"round": round_number, "gaps": len(gaps)},
        ) as observation:
            self._cover_by_reports(gaps, todos, reports)
            result.lookups = self._cover(
                gaps, todos, run_id, thread_id, allowed_source_ids, stored_text
            )
            open_now = {g.point_key for g in gaps if g.essential and g.resolution == "open"}
            open_before = {
                g["point_key"]
                for g in (previous or {}).get("gaps", [])
                if g.get("essential") and g.get("resolution") in {"open", "follow_up"}
            }
            analysis.closed_essential = len(open_before - open_now)
            if round_number >= 2:
                stall_count = 0 if analysis.closed_essential else stall_count + 1
            analysis.stall_count = stall_count
            follow_ups_so_far = [t for t in todos if t.origin == TaskOrigin.FOLLOW_UP]
            if not open_now:
                analysis.ended = "no_essential_gaps"
            elif stall_count >= STALL_LIMIT:
                analysis.ended = "stalled"
            elif len(follow_ups_so_far) >= self.max_per_run or self.router is None:
                analysis.ended = "caps_reached"
            else:
                result.follow_ups, result.replan = self._follow_ups(
                    gaps, checklist, follow_ups_so_far, round_number, run_id, thread_id, analysis
                )
                analysis.follow_ups = [t.todo_id for t in result.follow_ups]
                if result.follow_ups:
                    analysis.ended = "follow_ups_created"
                elif analysis.turn_limit_reached:
                    analysis.ended = "turn_limit"
                else:
                    analysis.ended = "caps_reached"
            for gap in gaps:
                if gap.essential and gap.resolution == "open" and analysis.ended != "follow_ups_created":
                    gap.resolution = "limitation"
            observation.output = {
                "ended": analysis.ended,
                "follow_ups": len(result.follow_ups),
                "closed_essential": analysis.closed_essential,
            }
        return result

    # ----- coverage across the run (FR-017a) -----

    @staticmethod
    def _cover_by_reports(gaps, todos, reports) -> None:
        """A gap is covered when a follow-up task that targeted it answered all its points."""
        for gap in gaps:
            if not gap.essential or gap.resolution != "open":
                continue
            for todo in todos:
                if gap.point_key not in todo.gap_point_ids:
                    continue
                report = reports.get(todo.todo_id)
                answered = [
                    p for p in (report.points if report else []) if p.status == PointStatus.ANSWERED
                ]
                if report and answered and len(answered) == len(report.points):
                    support = answered[0].support[0]
                    gap.resolution = "covered_by_report"
                    gap.covering_source_id = support.source_id
                    gap.quote = support.quote
                    gap.justification = f"Answered by follow-up {todo.todo_id}"
                    break

    def _cover(self, gaps, todos, run_id, thread_id, allowed, stored_text):
        essential = [g for g in gaps if g.essential and g.resolution == "open"]
        if not essential or self.router is None or self.lookup is None or not allowed:
            return []
        by_key = {
            f"{todo.todo_id}:{point.point_id}": (todo, point)
            for todo in todos
            for point in task_points(todo)
        }
        records, candidates = [], {}
        points_by_todo: dict[str, list] = {}
        for gap in essential:
            todo, point = by_key[gap.point_key]
            points_by_todo.setdefault(todo.todo_id, []).append((todo, point, gap))
        for items in points_by_todo.values():
            todo = items[0][0]
            phrasings, failed = phrasings_for(
                [point for _, point, _ in items],
                todo.research_question,
                [],
                self.router,
                run_id=run_id,
                thread_id=thread_id,
            )
            for _, point, gap in items:
                found = lookup_points(
                    [point],
                    todo=todo,
                    question=todo.research_question,
                    lookup=self.lookup,
                    run_id=run_id,
                    allowed_source_ids=allowed,
                    phrasings=phrasings,
                    phrasing_failed=failed,
                    requested_by="supervisor",
                )
                records.extend(found.records)
                if found.records:
                    gap.lookup_id = found.records[-1].lookup_id
                candidates[gap.point_key] = found.passages
        blocks = []
        for gap in essential:
            passages = candidates.get(gap.point_key) or {}
            text = "\n".join(
                f"  [{sid}] {chunk}" for sid, chunks in passages.items() for chunk in chunks
            ) or "  (no stored passage found)"
            blocks.append(f"{gap.point_key}: {gap.text}\n{text}")
        try:
            batch = self.router.invoke_messages(
                "supervisor",
                GAP_COVERAGE_TEMPLATE.format_messages(gaps="\n\n".join(blocks)),
                GapCoverageBatch,
                run_id=run_id,
                thread_id=thread_id,
            )
        except Exception:
            return records  # judgment unavailable: every essential gap stays open
        verdicts = {item.point_key: item for item in batch.gaps}
        for gap in essential:
            item = verdicts.get(gap.point_key)
            if item is not None:
                gap.justification = item.justification
            if not item or not item.covered or not item.source_id or not item.quote:
                continue
            if item.source_id not in allowed:
                continue
            haystack = "\n".join(candidates.get(gap.point_key, {}).get(item.source_id, []))
            haystack += "\n" + stored_text(item.source_id)
            if quote_in_text(item.quote, haystack):
                gap.resolution = "covered_by_lookup"
                gap.covering_source_id = item.source_id
                gap.quote = item.quote
        return records

    # ----- follow-up tasks (FR-017 to FR-020) -----

    def _follow_ups(self, gaps, checklist, existing, round_number, run_id, thread_id, analysis):
        open_gaps = {g.point_key: g for g in gaps if g.essential and g.resolution == "open"}
        created: list[ResearchTodo] = []
        replan = None
        per_answer: dict[str, int] = {}
        for todo in existing:
            answers = {a for p in todo.evidence_points for a in p.required_answers}
            for answer in answers:
                per_answer[answer] = per_answer.get(answer, 0) + 1
        seen_questions = {_normalized(t.research_question) for t in existing}
        messages = GAP_FOLLOW_UP_TEMPLATE.format_messages(
            round=round_number,
            remaining=self.max_per_run - len(existing),
            checklist="\n".join(f"{i}. {item}" for i, item in enumerate(checklist, start=1)),
            gaps="\n".join(
                f"- {g.point_key}: {g.text} (serves: {'; '.join(g.required_answers)})"
                for g in open_gaps.values()
            ),
        )
        tools = [CREATE_FOLLOW_UP_TOOL, PROPOSE_REPLAN_TOOL]
        for turn in range(1, MAX_TURNS + 1):
            analysis.turns_used = turn
            try:
                response = self.router.invoke_with_tools(
                    "supervisor", messages, tools, run_id=run_id, thread_id=thread_id, attempt=turn
                )
            except Exception as exc:
                analysis.refusals.append({"reason": f"model call failed: {redact_error(exc)}"})
                break
            if not response.tool_calls:
                break
            replies = []
            for call in response.tool_calls:
                args = call.get("args") or {}
                if call["name"] == PROPOSE_REPLAN_TOOL_NAME:
                    reply, proposal = self._replan(args)
                    replan = replan or proposal
                else:
                    reply = self._create(
                        call, args, open_gaps, checklist, per_answer, seen_questions,
                        existing, created, round_number, analysis,
                    )
                replies.append(ToolMessage(content=reply, tool_call_id=call["id"]))
            messages = [*messages, response, *replies]
            if replan is not None or not any(g.resolution == "open" for g in open_gaps.values()):
                break
            if len(existing) + len(created) >= self.max_per_run:
                break
        if analysis.turns_used >= MAX_TURNS:
            analysis.turn_limit_reached = True
        return created, replan

    @staticmethod
    def _replan(args: dict[str, Any]) -> tuple[str, dict | None]:
        reason = str(args.get("reason", "")).strip()
        todo_ids = [str(t) for t in args.get("todo_ids") or []]
        if not reason or not todo_ids:
            return "Refused: a replan needs a reason and the affected todo_ids.", None
        return "Replan proposed; a human will review a replacement plan.", {
            "reason": reason,
            "todo_ids": todo_ids,
        }

    def _create(
        self, call, args, open_gaps, checklist, per_answer, seen_questions,
        existing, created, round_number, analysis,
    ) -> str:
        if call["name"] != CREATE_FOLLOW_UP_TOOL_NAME:
            return f"Unknown tool {call['name']!r}."

        def refuse(text: str) -> str:
            analysis.refusals.append({"reason": text, "payload": args})
            return text

        question = str(args.get("question", "")).strip()
        keys = [str(k) for k in args.get("gap_point_keys") or []]
        if not question:
            return refuse("Refused: a follow-up needs a question.")
        bad = [k for k in keys if k not in open_gaps or open_gaps[k].resolution != "open"]
        if not keys or bad:
            return refuse(f"Refused: {', '.join(bad) or 'gap_point_keys'} is unknown or already covered.")
        try:
            drafts = [EvidencePointDraft.model_validate(p) for p in args.get("evidence_points") or []]
            if not 1 <= len(drafts) <= 4 or any(not d.required_answers for d in drafts):
                raise ValueError("1 to 4 evidence points, each linked to a required answer")
            points = evidence_points_from_drafts(drafts, checklist)
        except (ValidationError, ValueError) as exc:
            return refuse(
                f"Refused: evidence points must link to required answers 1..{len(checklist)} ({exc})."
            )
        answers = {a for p in points for a in p.required_answers}
        over = [a for a in answers if per_answer.get(a, 0) >= self.max_per_answer]
        if over:
            return refuse(f"Refused: follow-up limit reached for {over[0]}.")
        if len(existing) + len(created) >= self.max_per_run:
            return refuse("Refused: follow-up limit reached for this run.")
        normalized = _normalized(question)
        if normalized in seen_questions:
            duplicate = next(
                (t.todo_id for t in [*existing, *created]
                 if _normalized(t.research_question) == normalized),
                "an earlier follow-up",
            )
            return refuse(f"Refused: duplicates follow-up {duplicate}.")
        todo = ResearchTodo(
            research_question=question,
            evidence_points=points,
            evidence_needed="; ".join(p.text for p in points),
            origin=TaskOrigin.FOLLOW_UP,
            round=round_number + 1,
            gap_point_ids=keys,
        )
        created.append(todo)
        seen_questions.add(normalized)
        for answer in answers:
            per_answer[answer] = per_answer.get(answer, 0) + 1
        for key in keys:
            open_gaps[key].resolution = "follow_up"
        return f"Created follow-up {todo.todo_id}."
