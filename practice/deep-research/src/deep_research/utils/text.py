"""Shared prompt formatting and citation patterns."""

from __future__ import annotations

import re
from datetime import datetime

CITATION = re.compile(r"\[(\d+)\]")


SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def today() -> str:
    """Return the local date in the format used by research prompts.

    Returns:
        Date formatted as abbreviated weekday, month, day, and year.
    """
    return datetime.now().strftime("%a %b %d, %Y")


def bullets(items: list[str]) -> str:
    """Format strings as Markdown bullet lines.

    Args:
        items: Items to render without altering their contents.

    Returns:
        Bullet list, or an empty string for no items.
    """
    return "\n".join(f"- {item}" for item in items)
