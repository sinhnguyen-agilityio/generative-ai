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
    REFLECTION_TEMPLATE,
    REPORT_WRITER_TEMPLATE,
    REVISION_BLOCK,
)
from intelligence_collector.contracts import (
    DraftAssessment,
    Reflection,
    ResearchBrief,
    VerificationResult,
)
from intelligence_collector.source_verification import EvidenceBundle, SourceCatalog
from intelligence_collector.tools.url_tracking import normalize_url

EXCERPT_CHARS = 1200
MAX_LESSONS = 12
DEFAULT_EVIDENCE_TOP_K = 6


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


def select_relevant_evidence(
    catalog: SourceCatalog,
    index,
    run_id: str,
    *,
    top_k: int = DEFAULT_EVIDENCE_TOP_K,
) -> SourceCatalog:
    """Narrow each bundle to its top-K most relevant sources via vector retrieval.

    Scoped *per research question* (bundle), never globally, so every question keeps
    guaranteed representation in the Build/Fix prompt — a global top-K could let one
    question's sources crowd out another's entirely, which would make the report unable to
    cite that question and fail the verifier's ``question_coverage`` check for a reason that
    has nothing to do with missing evidence.

    Verification always reads the full, unfiltered catalog independently (``BVFNodes._catalog``
    calls ``load_catalog`` directly); this narrowing affects only what the writer's prompt sees,
    never what a claim is checked against. A bundle at or under ``top_k`` sources is left as-is,
    and retrieval failures never block Build — they just leave the bundle unfiltered.
    """
    if index is None:
        return catalog
    narrowed = []
    for bundle in catalog.bundles:
        if len(bundle.source_ids) <= top_k:
            narrowed.append(bundle)
            continue
        try:
            chunks = index.retrieve(
                bundle.question, limit=top_k, run_id=run_id, todo_id=bundle.todo_id
            )
        except Exception:
            narrowed.append(bundle)
            continue
        url_to_sid = {
            normalize_url(catalog.sources[sid].url): sid for sid in bundle.source_ids
        }
        ranked: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            sid = url_to_sid.get(normalize_url(chunk.metadata.get("locator", "")))
            if sid and sid not in seen:
                seen.add(sid)
                ranked.append(sid)
        for sid in bundle.source_ids:  # keep exactly top_k even if retrieval matched fewer
            if len(ranked) >= top_k:
                break
            if sid not in seen:
                seen.add(sid)
                ranked.append(sid)
        narrowed.append(EvidenceBundle(
            todo_id=bundle.todo_id,
            question=bundle.question,
            summary=bundle.summary,
            source_ids=ranked,
            gaps=bundle.gaps,
        ))
    return SourceCatalog(sources=catalog.sources, bundles=narrowed)


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
            run_id=run_id,
            thread_id=thread_id,
        )
