"""Normalize human review decisions and edited items."""

from __future__ import annotations


def _feedback(decision: dict, default: str) -> str:
    """Resolve a human decision into revision feedback.

    Args:
        decision: Review response with approval and optional feedback.
        default: Revision request used when approval and feedback are both absent.

    Returns:
        Empty text for approval; otherwise supplied feedback or the default.
    """
    return "" if decision.get("approved") else decision.get("feedback") or default


def _lines(items: list[str]) -> list[str]:
    """Strip review items and discard blank entries.

    Args:
        items: Edited topics or acceptance criteria.

    Returns:
        Nonempty items in their original order.
    """
    return [item.strip() for item in items if item.strip()]
