"""Helpers shared by the agents."""

from __future__ import annotations

import re
from datetime import datetime

from langchain_core.runnables import RunnableConfig

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


def thread_id(config: RunnableConfig) -> str:
    """Read the research thread identifier from graph configuration.

    Args:
        config: Runnable configuration containing configurable.thread_id.

    Returns:
        Thread identifier used for storage and tracing.
    """
    return config["configurable"]["thread_id"]


def merge_notes(notes: list[dict]) -> tuple[str, dict[int, tuple[str, str]]]:
    """Join topic notes and assign one citation number per URL across the run.

    Rewrite local citations from source metadata rather than parsing source lists from
    report text. Unknown local citation numbers are removed.

    Args:
        notes: Topic notes with report text and local source registries.

    Returns:
        Joined reports and a mapping from global source number to (title, URL).
    """
    numbers: dict[str, int] = {}
    sources: dict[int, tuple[str, str]] = {}
    merged = []
    for note in notes:
        local = {}
        for source in note["sources"]:
            local[source["id"]] = numbers.setdefault(source["url"], len(numbers) + 1)
            sources.setdefault(local[source["id"]], (source["title"], source["url"]))
        merged.append(CITATION.sub(lambda m: f"[{local[int(m.group(1))]}]" if int(m.group(1)) in local else "", note["report"]))
    return "\n\n---\n\n".join(merged), sources


def passages_by_url(notes: list[dict]) -> dict[str, list[str]]:
    """Collect unique source passages from topic notes, grouped by URL.

    Args:
        notes: Topic notes carrying cited source metadata and passages.

    Returns:
        URL-to-passages mapping preserving first-seen order.
    """
    passages: dict[str, dict[str, None]] = {}
    for note in notes:
        for source in note["sources"]:
            passages.setdefault(source["url"], {}).update(dict.fromkeys(source.get("passages", [])))
    return {url: list(texts) for url, texts in passages.items()}
