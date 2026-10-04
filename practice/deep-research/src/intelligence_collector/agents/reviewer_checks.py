"""Executable reviewer constraints; no report truncation or LLM-based counting."""

import re

from intelligence_collector.contracts import ReportJudgeCriterion


def effective_requests(requests: list[str]) -> list[str]:
    """Supersede only standalone word limits; preserve mixed semantic instructions."""
    latest = max(
        (index for index, request in enumerate(requests) if word_constraint(request) is not None),
        default=-1,
    )
    result = []
    for index, request in enumerate(requests):
        simple = re.fullmatch(
            r"\s*(?:please\s+)?(?:write\s+(?:the\s+)?(?:report\s+)?(?:with\s+)?)?"
            r"(?:(?:exactly|at most|no more than|under|at least)\s+)?\d+\s+words?"
            r"(?:\s+only)?[.!]?\s*",
            request,
            re.IGNORECASE,
        )
        if index == latest or simple is None or word_constraint(request) is None:
            result.append(request)
    return result


def word_constraint(request: str) -> tuple[str, int] | None:
    """Recognize explicit English word limits; leave ambiguous requests to the judge."""
    match = re.search(
        r"\b(?:(exactly|at most|no more than|under|at least)\s+)?(\d+)\s+words?\b",
        request,
        re.IGNORECASE,
    )
    if not match:
        return None
    mode = (match[1] or "exactly").lower()
    return mode, int(match[2])


def reviewer_format_checks(report: str, requests: list[str]) -> list[ReportJudgeCriterion]:
    """Count whitespace-separated body tokens, excluding Markdown Sources and citations.

    The latest explicit word instruction supersedes previous word limits. Other requests
    remain active. Heading text counts; punctuation alone and citation markers do not.
    """
    limits = [word_constraint(request) for request in requests]
    limit = next((item for item in reversed(limits) if item is not None), None)
    if limit is None:
        return []
    body = re.split(r"(?im)^\s*#{1,6}\s+Sources\s*$", report, maxsplit=1)[0]
    body = re.sub(r"\[S\d+\]", "", body)
    count = sum(bool(re.search(r"\w", token)) for token in body.split())
    mode, target = limit
    met = (
        count <= target
        if mode in {"at most", "no more than"}
        else count < target
        if mode == "under"
        else count >= target
        if mode == "at least"
        else count == target
    )
    return [
        ReportJudgeCriterion(
            criterion=f"Reviewer word limit: {mode} {target} words (body only).",
            met=met,
            justification=f"Measured {count} body words; Sources and citation markers excluded.",
        )
    ]
