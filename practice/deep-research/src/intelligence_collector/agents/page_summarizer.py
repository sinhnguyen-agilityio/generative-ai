"""Page summarizer: turns each fetched page into a short, topic-focused summary.

Tavily returns every page's full text (``raw_content``). Putting whole pages into the researcher's
context would overflow it, so each usable page is cleaned and summarized by a cheap model
(``page_summarizer`` role) into ``PageSummary`` — a summary plus verbatim key excerpts. The
summary becomes the result's ``content`` (what the researcher, compressor and report writer read);
the cleaned page stays in ``raw_content`` for the VFS and the vector index only.

A quote the model did not copy word for word from the page is dropped, so excerpts are always real
page text. A page that is too short, or a summarizer failure, keeps Tavily's snippet instead.
"""

from __future__ import annotations

import contextvars
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Any

from intelligence_collector.agents.prompts import PAGE_SUMMARY_TEMPLATE
from intelligence_collector.contracts import PageSummary
from intelligence_collector.models.redaction import redact_text
from intelligence_collector.tools.page_content import clean_page, usable_page

# Page text sent to the summarizer (about 10k tokens); the stored page may be longer.
SUMMARIZER_INPUT_CHARS = 40_000
MAX_PARALLEL_SUMMARIES = 3

_SPACES = re.compile(r"\s+")


def _squash(text: str) -> str:
    return _SPACES.sub(" ", text).strip().lower()


def verbatim_excerpts(excerpts: list[str], page: str) -> list[str]:
    """Keep only the quotes that appear in the page (ignoring whitespace and case)."""
    haystack = _squash(page)
    return [quote for quote in excerpts if _squash(quote) and _squash(quote) in haystack]


def format_page_evidence(summary: PageSummary) -> str:
    """The text the researcher and the report writer read for one summarized page."""
    if not summary.key_excerpts:
        return summary.summary
    quotes = "\n".join(f'- "{quote}"' for quote in summary.key_excerpts)
    return f"{summary.summary}\n\nKey excerpts:\n{quotes}"


class PageSummarizer:
    """Summarizes fetched pages for one research topic with the ``page_summarizer`` model role."""

    role = "page_summarizer"

    def __init__(self, router):
        self.router = router

    def summarize(
        self, topic: str, *, url: str, title: str, page: str, run_id: str = "", thread_id: str = ""
    ) -> PageSummary:
        """One page's summary; excerpts not found verbatim in the page are removed."""
        messages = PAGE_SUMMARY_TEMPLATE.format_messages(
            date=date.today().isoformat(),
            topic=topic,
            title=title or "Untitled",
            url=url,
            page=page[:SUMMARIZER_INPUT_CHARS],
        )
        summary = self.router.invoke_messages(
            self.role, messages, PageSummary, run_id=run_id, thread_id=thread_id
        )
        return summary.model_copy(
            update={"key_excerpts": verbatim_excerpts(summary.key_excerpts, page)}
        )

    def read_pages(
        self, topic: str, rows: list[dict[str, Any]], *, run_id: str = "", thread_id: str = ""
    ) -> list[dict[str, Any]]:
        """Clean each row's raw page and replace its snippet ``content`` with the page summary.

        Rows keep their order. ``raw_content`` becomes the cleaned page, or "" when the page is
        unusable; ``snippet`` keeps Tavily's original excerpt whenever a summary replaced it.
        """

        def read(row: dict[str, Any]) -> dict[str, Any]:
            page, truncated = clean_page(str(row.get("raw_content") or ""))
            if not usable_page(page):
                return {**row, "raw_content": ""}
            row = {**row, "raw_content": page, "raw_truncated": truncated}
            try:
                summary = self.summarize(
                    topic,
                    url=str(row.get("url", "")),
                    title=str(row.get("title", "")),
                    page=page,
                    run_id=run_id,
                    thread_id=thread_id,
                )
            except Exception as exc:  # the snippet is still usable evidence
                return {**row, "summary_error": redact_text(f"{type(exc).__name__}: {exc}")}
            return {
                **row,
                "content": format_page_evidence(summary),
                "snippet": str(row.get("content", "")),
            }

        if len(rows) <= 1:
            return [read(row) for row in rows]
        # Each worker thread runs in a copy of the caller's context, so model-call traces stay
        # nested under the current research span.
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SUMMARIES) as pool:
            futures = [pool.submit(contextvars.copy_context().run, read, row) for row in rows]
            return [future.result() for future in futures]
