"""Report writer for the Build-Verify-Fix loop.

* ``write``   Build (first draft) or Fix (revision) of the report from stored evidence. A fix
              attempt sees the failed checks, every lesson learned so far, the planned strategy,
              and the previous draft.
* ``reflect`` Self-improvement: diagnose why verification failed and turn it into reusable
              lessons and a strategy, instead of blindly retrying the same generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date

from intelligence_collector.agents.prompts import (
    REFLECTION_PROMPT_VERSION,
    REFLECTION_TEMPLATE,
    REPORT_WRITER_PROMPT_VERSION,
    REPORT_WRITER_TEMPLATE,
    REVISION_BLOCK,
)
from intelligence_collector.contracts import (
    DraftAssessment,
    Reflection,
    ResearchBrief,
    VerificationResult,
)
from intelligence_collector.source_verification import SourceCatalog

EXCERPT_CHARS = 1200
MAX_LESSONS = 12


@dataclass
class RevisionContext:
    """What a Fix attempt must learn from."""

    attempt: int
    failures: str
    previous_draft: str
    lessons: list[str] = field(default_factory=list)
    strategy: str = ""


def _bullets(items: list[str], empty: str = "(none)") -> str:
    return "\n".join(f"- {item}" for item in items) if items else empty


def format_evidence(catalog: SourceCatalog) -> str:
    blocks = []
    for bundle in catalog.bundles:
        sources = [
            f"- [{sid}] {catalog.sources[sid].title or 'Untitled'} - {catalog.sources[sid].url}\n"
            f"  Excerpt: {catalog.sources[sid].excerpt[:EXCERPT_CHARS]}"
            for sid in bundle.source_ids
        ]
        blocks.append(
            f"### Research question: {bundle.question}\n"
            f"Findings summary (URLs map to the source ids below):\n{bundle.summary or '(none)'}\n"
            f"Sources:\n" + ("\n".join(sources) or "(none)")
        )
    return "\n\n".join(blocks) or "(no evidence)"


def merge_lessons(existing: list[str], new: list[str]) -> list[str]:
    """Accumulate lessons across attempts, newest last, without duplicates."""
    merged: dict[str, str] = {}
    for lesson in [*existing, *new]:
        key = lesson.strip().casefold()
        if key:
            merged.pop(key, None)
            merged[key] = lesson.strip()
    return list(merged.values())[-MAX_LESSONS:]


class ReportWriter:
    def __init__(self, router):
        self.router = router

    def write(
        self,
        brief: ResearchBrief,
        catalog: SourceCatalog,
        *,
        revision: RevisionContext | None = None,
        run_id: str = "",
        thread_id: str = "",
    ) -> str:
        revision_text = ""
        if revision is not None:
            revision_text = REVISION_BLOCK.format(
                attempt=revision.attempt,
                failures=revision.failures,
                lessons=_bullets(revision.lessons),
                strategy=revision.strategy or "(fix every failed check)",
                previous_draft=revision.previous_draft,
            )
        messages = REPORT_WRITER_TEMPLATE.format_messages(
            date=date.today().isoformat(),
            brief=brief.research_question,
            evidence=format_evidence(catalog),
            gaps=_bullets([f"{gap.text} (question: {gap.question})" for gap in catalog.gaps]),
            revision=revision_text,
        )
        draft = self.router.invoke_messages(
            "synthesizer",
            messages,
            DraftAssessment,
            prompt_version=REPORT_WRITER_PROMPT_VERSION,
            run_id=run_id,
            thread_id=thread_id,
        )
        return draft.report_markdown

    def reflect(
        self,
        draft: str,
        verification: VerificationResult,
        *,
        history: list[dict],
        lessons: list[str],
        stalled: bool,
        run_id: str = "",
        thread_id: str = "",
    ) -> Reflection:
        previous = [
            {key: entry.get(key) for key in ("attempt", "failed_checks", "diagnosis", "strategy")}
            for entry in history
        ]
        messages = REFLECTION_TEMPLATE.format_messages(
            failures=verification.repair_details,
            history=json.dumps(previous, indent=2) if previous else "(first failure)",
            lessons=_bullets(lessons),
            stalled="yes" if stalled else "no",
            draft=draft,
        )
        return self.router.invoke_messages(
            "fixer",
            messages,
            Reflection,
            prompt_version=REFLECTION_PROMPT_VERSION,
            run_id=run_id,
            thread_id=thread_id,
        )
