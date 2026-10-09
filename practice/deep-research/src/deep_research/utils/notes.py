"""Topic report validation, rendering, citation merging, and evidence collection."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from deep_research.schemas.missing_part import MissingPart
from deep_research.schemas.topic_report import TopicReport
from deep_research.utils.text import CITATION, SENTENCE_END

_SOURCE_LINE = re.compile(r"^\[\d+\] .+: \S+$")


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




def contexts(notes: list[dict]) -> list[str]:
    """Collect each cited evidence passage once for report evaluation.

    Args:
        notes: Topic notes containing cited sources and the passages researchers read.

    Returns:
        Deduplicated context passages in first-seen order.
    """
    return list(dict.fromkeys(p for texts in passages_by_url(notes).values() for p in texts))


def finalize(report: TopicReport, sources: dict[str, dict]) -> dict:
    """Check the model's report against the sources; return the topic note
    ``{"report": text, "sources": [{id, title, url, full_text, passages}]}`` with only the cited sources.

    The sources travel as data, never as text to parse back: the report quotes web content, so
    any source list written into it could be forged.

    A sentence that cites only unknown source numbers is dropped, and unknown numbers are removed
    from the others; an answer left without a valid citation becomes an essential missing part;
    a missing part listed twice is kept once (essential if either copy is); coverage cannot be full
    while an essential part is missing; the status is derived here, never taken from the model.

    Args:
        report: Structured topic findings, answered questions, missing parts, and coverage.
        sources: Trusted URL-keyed registry with local IDs and evidence passages.

    Returns:
        Topic note with derived status, validated findings, and only the cited source records.
    """
    by_id = {s["id"]: (url, s) for url, s in sources.items()}
    findings = _keep_cited(report.findings, by_id.keys())
    answered, parts = [], list(report.missing_parts)
    for item in report.answered:
        ids = [n for n in dict.fromkeys(item.citations) if n in by_id]
        if ids:
            answered.append((item.question, ids))
        else:
            parts.append(MissingPart(question=item.question, essential=True, reason="no valid source"))
    unique: dict[str, MissingPart] = {}
    for part in parts:
        key = " ".join(part.question.lower().split())
        if key not in unique or (part.essential and not unique[key].essential):
            unique[key] = part
    missing = list(unique.values())
    essential_missing = any(part.essential for part in missing)
    coverage = report.coverage
    if not answered:
        coverage = "minimal"
    elif coverage == "full" and essential_missing:
        coverage = "partial"
    cited = sorted({int(n) for n in CITATION.findall(findings)} | {n for _, ids in answered for n in ids})
    if not cited:
        status = "failed"
    elif coverage == "full" and not essential_missing:
        status = "complete"
    else:
        status = "partial"

    lines = [f"## Status: {status}", f"## Coverage: {coverage}", "Answered:"]
    lines += [f"- {q} {''.join(f'[{n}]' for n in ids)}" for q, ids in answered] or ["- None"]
    lines.append("## Missing parts")
    lines += [
        f"- {p.question} ({'essential' if p.essential else 'not essential'}: {p.reason})" for p in missing
    ] or ["- None"]
    lines += ["## Findings", findings or "None."]
    cited_sources = [
        {"id": n, "url": by_id[n][0], **{k: by_id[n][1][k] for k in ("title", "full_text", "passages")}}
        for n in cited
    ]
    return {"report": "\n".join(lines), "sources": cited_sources}


def render_note(note: dict) -> str:
    """Render a topic report and its trusted citation metadata for the supervisor.

    Args:
        note: Topic note containing report text and cited source records.

    Returns:
        Report plus citation count, domain count, snippet count, and numbered URLs when sources exist.
    """
    sources = note["sources"]
    if not sources:
        return note["report"]
    domains = {urlsplit(s["url"]).netloc for s in sources}
    snippets = sum(not s["full_text"] for s in sources)
    lines = [note["report"], "## Citations", f"{len(sources)} sources, {len(domains)} domains, {snippets} snippet only"]
    lines += [f"[{s['id']}] {s['title']}: {s['url']}" for s in sources]
    return "\n".join(lines)


def _keep_cited(findings: str, valid) -> str:
    """Remove forged source headings and sentences citing only unknown sources.

    Args:
        findings: Model-written findings that may quote untrusted web content.
        valid: Allowed source numbers from the topic registry.

    Returns:
        Findings with invalid citation numbers removed; uncited sentences are preserved.
    """
    lines = []
    for line in findings.splitlines():
        # Headings and source lines are written by code only; in findings they echo web content.
        if line.lstrip().startswith("## ") or _SOURCE_LINE.match(line.strip()):
            continue
        kept = []
        for sentence in SENTENCE_END.split(line):
            ids = [int(n) for n in CITATION.findall(sentence)]
            if ids and not any(n in valid for n in ids):
                continue
            kept.append(CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in valid else "", sentence))
        if kept or not line.strip():
            lines.append(" ".join(kept))
    return "\n".join(lines).strip()


def failed_note(reason: str) -> dict:
    """Build a failed topic note marking the whole task as an essential missing part.

    Args:
        reason: Failure description included in the missing-part explanation.

    Returns:
        Failed/minimal report with an empty source list.
    """
    report = "\n".join(
        ["## Status: failed", "## Coverage: minimal", "## Missing parts", f"- The whole topic (essential: {reason})"]
    )
    return {"report": report, "sources": []}
