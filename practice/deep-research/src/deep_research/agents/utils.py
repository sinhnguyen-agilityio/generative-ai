"""Helpers shared by the agents."""

from __future__ import annotations

import re
from datetime import datetime

from langchain_core.runnables import RunnableConfig

CITATION = re.compile(r"\[(\d+)\]")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def today() -> str:
    return datetime.now().strftime("%a %b %d, %Y")


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def thread_id(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]


def merge_notes(notes: list[dict]) -> tuple[str, dict[int, tuple[str, str]]]:
    """Join the topic notes with one numbering for the whole run.

    Every topic note numbers its sources from [1]; here each URL gets one run-wide number and the
    note's citations are rewritten to it. The numbers come from each note's ``sources`` data, never
    from its text. Returns the joined reports and ``{number: (title, url)}``."""
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
    """The passages the researchers were shown, per cited URL, each passage once."""
    passages: dict[str, dict[str, None]] = {}
    for note in notes:
        for source in note["sources"]:
            passages.setdefault(source["url"], {}).update(dict.fromkeys(source.get("passages", [])))
    return {url: list(texts) for url, texts in passages.items()}
