"""Result evaluation: a searcher accepts or rejects every search result it is shown (FR-005).

Cheap deterministic rules run first: a page this task already evaluated is a ``duplicate``; a
row with neither a readable page nor a snippet has ``no_usable_content``. The remaining rows go
to one structured call (role ``result_evaluator``) that accepts a row or rejects it as off-topic,
low-credibility or outside the task's scope, and names the evidence points it covers. Code
validates the answer: a row the model omits is rejected as off-topic and recorded as a
correction. When the call fails, nothing from that search is accepted and the caller counts the
search as failed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from intelligence_collector.agents.prompts import RESULT_EVALUATOR_TEMPLATE
from intelligence_collector.contracts import (
    RejectReason,
    ResearchTodo,
    ResultEvaluation,
    ResultEvaluationBatch,
)
from intelligence_collector.models.redaction import redact_error
from intelligence_collector.observability.tracing import traced
from intelligence_collector.ports import ModelRouterPort
from intelligence_collector.tools.url_tracking import normalize_url

ROW_PASSAGE_CHARS = 1600


@dataclass
class EvaluationOutcome:
    """Decisions for one search, in row order, plus what went wrong."""

    evaluations: list[ResultEvaluation] = field(default_factory=list)
    corrections: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.error)


def _has_content(row: dict[str, Any]) -> bool:
    return bool(str(row.get("raw_content") or "").strip() or str(row.get("content") or "").strip())


def _points_block(todo: ResearchTodo) -> str:
    return "\n".join(f"- {p.point_id}: {p.text}" for p in todo.evidence_points) or "(none listed)"


def _rows_block(rows: list[tuple[int, dict[str, Any]]]) -> str:
    return "\n\n".join(
        f"[{number}] {row.get('title') or 'Untitled'}\nURL: {row.get('url', '')}\n"
        f"{str(row.get('content') or '')[:ROW_PASSAGE_CHARS]}"
        for number, row in rows
    )


def evaluate_results(
    rows: list[dict[str, Any]],
    *,
    todo: ResearchTodo,
    question: str,
    query: str,
    search_index: int,
    seen_urls: set[str],
    router: ModelRouterPort,
    run_id: str = "",
    thread_id: str = "",
) -> EvaluationOutcome:
    """Decide every row of one search, in row order. `seen_urls` holds the normalized URLs this
    task evaluated in earlier searches; it is not modified here."""
    outcome = EvaluationOutcome()
    decisions: list[ResultEvaluation | None] = []
    pending: list[tuple[int, int, dict[str, Any]]] = []  # (number shown, row position, row)
    batch_urls: set[str] = set()
    point_ids = {p.point_id for p in todo.evidence_points}

    def rejected(row: dict[str, Any], reason: RejectReason) -> ResultEvaluation:
        return ResultEvaluation(
            url=str(row.get("url", "")),
            title=str(row.get("title", "")),
            query=query,
            search_index=search_index,
            decision="rejected",
            reason=reason,
        )

    for position, row in enumerate(rows):
        key = normalize_url(str(row.get("url", "")))
        if key in seen_urls or key in batch_urls:
            decisions.append(rejected(row, RejectReason.DUPLICATE))
        elif not _has_content(row):
            decisions.append(rejected(row, RejectReason.NO_USABLE_CONTENT))
        else:
            pending.append((len(pending) + 1, position, row))
            decisions.append(None)
        batch_urls.add(key)

    if pending:
        batch: ResultEvaluationBatch | None = None
        try:
            with traced(
                "result_evaluation",
                metadata={"todo_id": todo.todo_id, "rows": len(pending)},
            ) as observation:
                batch = router.invoke_messages(
                    "result_evaluator",
                    RESULT_EVALUATOR_TEMPLATE.format_messages(
                        question=question,
                        points=_points_block(todo),
                        rows=_rows_block([(number, row) for number, _, row in pending]),
                    ),
                    ResultEvaluationBatch,
                    run_id=run_id,
                    thread_id=thread_id,
                )
                observation.output = {
                    "accepted": sum(item.decision == "accepted" for item in batch.rows)
                }
        except Exception as exc:
            outcome.error = f"Result evaluation failed: {redact_error(exc)}"
        answered = {}
        for item in batch.rows if batch is not None else []:
            answered.setdefault(item.index, item)
        for number, position, row in pending:
            item = answered.get(number)
            if batch is None:
                decisions[position] = rejected(row, RejectReason.OFF_TOPIC)
            elif item is None:
                outcome.corrections.append(
                    f"Result {number} ({row.get('url', '')}) was not evaluated; "
                    "rejected as off_topic"
                )
                decisions[position] = rejected(row, RejectReason.OFF_TOPIC)
            elif item.decision == "accepted":
                decisions[position] = ResultEvaluation(
                    url=str(row.get("url", "")),
                    title=str(row.get("title", "")),
                    query=query,
                    search_index=search_index,
                    decision="accepted",
                    covers=[c for c in dict.fromkeys(item.covers) if c in point_ids],
                    snippet_only=not str(row.get("raw_content") or "").strip(),
                )
            else:
                decisions[position] = rejected(
                    row, RejectReason(item.reason) if item.reason else RejectReason.OFF_TOPIC
                )
    outcome.evaluations = [decision for decision in decisions if decision is not None]
    return outcome
