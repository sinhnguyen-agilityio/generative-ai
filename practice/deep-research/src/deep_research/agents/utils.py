"""Helpers shared by the agents."""

from __future__ import annotations

import re
from datetime import datetime

from langchain_core.runnables import RunnableConfig

CITATION = re.compile(r"\[(\d+)\]")
SOURCE_LINE = re.compile(r"^\[(\d+)\] (.+): (\S+)$", re.M)
CITATIONS_HEADING = "## Citations"


def today() -> str:
    return datetime.now().strftime("%a %b %d, %Y")


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def thread_id(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


def merge_notes(notes: list[str]) -> tuple[str, dict[int, tuple[str, str]]]:
    """Join the topic reports with one numbering for the whole run.

    Every topic report numbers its sources from [1]; here each URL gets one run-wide number and
    the report's citations are rewritten to it. Returns the joined reports, without their own
    Citations sections, and ``{number: (title, url)}``. Only the last Citations section is read:
    ``finalize`` writes it last, and anything before it may quote web content."""
    numbers: dict[str, int] = {}
    sources: dict[int, tuple[str, str]] = {}
    merged = []
    for note in notes:
        body, heading, citations = note.rpartition(CITATIONS_HEADING)
        if not heading:
            body, citations = note, ""
        local = {}
        for n, title, url in SOURCE_LINE.findall(citations):
            local[int(n)] = numbers.setdefault(url, len(numbers) + 1)
            sources.setdefault(local[int(n)], (title, url))
        merged.append(CITATION.sub(lambda m: f"[{local[int(m.group(1))]}]" if int(m.group(1)) in local else "", body).strip())
    return "\n\n---\n\n".join(merged), sources
