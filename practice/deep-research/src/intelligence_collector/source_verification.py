"""Evidence catalog shared by the report writer, the LLM judge and RAGAS scoring.

Loads the evidence bundles the research sub-agents stored for a run and gives every unique source
URL a stable ``S<n>`` id, which is what reports cite (``[S1]``). Also records the sub-agents'
unresolved evidence gaps (disclosed by the writer), and fingerprints a verification failure so the Build-Verify-Fix loop can
tell when a fix made no progress.

Structural citation checks complement, but never replace, ``agents/report_judge.py``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field

from intelligence_collector.contracts import VerificationResult
from pathlib import Path
from intelligence_collector.tools.url_tracking import normalize_url

CATALOG_STATE_PATH = "verification/catalog_state.json"

_CITATION = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")


# ===== EVIDENCE CATALOG =====


@dataclass(frozen=True)
class CatalogSource:
    """One unique source page, as the report cites it."""

    source_id: str  # "S1", "S2", ... stable per unique URL
    url: str
    title: str
    excerpt: str


@dataclass(frozen=True)
class EvidenceGap:
    """Something a research sub-agent could not find evidence for (its `unresolved_gaps`); the
    writer discloses each one under ``## Limitations``."""

    todo_id: str
    question: str
    text: str


@dataclass
class EvidenceBundle:
    """The evidence one research sub-agent gathered for one research question."""

    todo_id: str
    question: str
    summary: str
    source_ids: list[str]
    gaps: list[EvidenceGap]


@dataclass
class SourceCatalog:
    """Every source of a run under its ``S<n>`` id, plus the per-question bundles."""

    sources: dict[str, CatalogSource] = field(default_factory=dict)
    bundles: list[EvidenceBundle] = field(default_factory=list)

    @property
    def gaps(self) -> list[EvidenceGap]:
        """Unresolved gaps across all bundles."""
        return [gap for bundle in self.bundles for gap in bundle.gaps]


def _normalize_url(url: str) -> str:
    """Normalize a source URL so the same page always maps to the same ``S<n>`` id."""
    return normalize_url(url.strip().rstrip(".,;:"))


def catalog_from_bundles(bundles: list[dict]) -> SourceCatalog:
    """Assign stable ``S<n>`` ids (first appearance order) to every unique source URL."""
    catalog = SourceCatalog()
    by_url: dict[str, str] = {}
    for bundle in bundles:
        todo_id = str(bundle.get("task_id", ""))
        question = str(bundle.get("question", ""))
        source_ids = []
        for source in bundle.get("sources", []):
            url = str(source.get("locator") or source.get("url") or "").strip()
            if not url:
                continue
            key = _normalize_url(url)
            if key not in by_url:
                by_url[key] = f"S{len(by_url) + 1}"
                provenance = source.get("provenance") or {}
                catalog.sources[by_url[key]] = CatalogSource(
                    source_id=by_url[key],
                    url=url,
                    title=str(provenance.get("title")
                              or source.get("title") or ""),
                    excerpt=str(source.get("excerpt")
                                or source.get("content") or ""),
                )
            if by_url[key] not in source_ids:
                source_ids.append(by_url[key])
        gaps = [
            EvidenceGap(todo_id, question, str(text).strip())
            for text in bundle.get("unresolved_gaps", [])
            if str(text).strip()
        ]
        catalog.bundles.append(EvidenceBundle(
            todo_id=todo_id,
            question=question,
            summary=str(bundle.get("summary", "")),
            source_ids=source_ids,
            gaps=gaps,
        ))
    return catalog


def cited_source_excerpts(report: str, catalog: SourceCatalog) -> list[str]:
    """Excerpt text of every source id actually cited in the report body.

    This is the RAGAS Faithfulness "retrieved_contexts" for a whole-report evaluation: the
    evidence the report was supposed to stay faithful to is exactly what it cites, not the
    full (larger, partly unused) catalog the run gathered.
    """
    return [
        catalog.sources[sid].excerpt
        for sid in dict.fromkeys(_cited_ids(report))
        if sid in catalog.sources and catalog.sources[sid].excerpt
    ]


def evidence_task_ids(evidence_refs: list[dict]) -> list[str]:
    """Task ids of the evidence bundles in graph state, deduplicated in research order."""
    return list(dict.fromkeys(ref["task_id"] for ref in evidence_refs if ref.get("task_id")))


def load_catalog(vfs, run_id: str, evidence_refs: list[dict], *, active_todo_ids=None) -> SourceCatalog:
    """Read the evidence bundles referenced by graph state (in research order)."""
    bundles = []
    seen = set()
    base = (vfs.root / run_id).resolve()
    for ref in evidence_refs:
        task_id = ref.get("task_id")
        if not task_id:
            continue
        path = ref.get("path")
        relative = str(Path(path).resolve().relative_to(base)) if path else (
            f"research/{task_id}/evidence/result.json"
        )
        if relative in seen:
            continue
        seen.add(relative)
        bundles.append(json.loads(vfs.read(run_id, relative)))
    # Keep historic sources in their original order for stable citations, but only the newest
    # attempt determines each todo's summary and unresolved gaps.
    latest = {}
    all_sources = {}
    for bundle in bundles:
        latest[bundle.get("task_id")] = bundle
        all_sources.setdefault(bundle.get("task_id"), []).extend(bundle.get("sources", []))
    # Seed citations from capture order before combining the latest per-question findings.
    historic = catalog_from_bundles(bundles)
    merged = catalog_from_bundles([
        {**bundle, "sources": all_sources[task_id]} for task_id, bundle in latest.items()
    ])
    by_url = {_normalize_url(source.url): sid for sid, source in historic.sources.items()}
    for bundle in merged.bundles:
        bundle.source_ids = [by_url[_normalize_url(merged.sources[sid].url)]
                             for sid in bundle.source_ids]
    merged.sources = historic.sources
    if active_todo_ids is not None:
        active = set(active_todo_ids)
        merged.bundles = [bundle for bundle in merged.bundles if bundle.todo_id in active]
        used = {sid for bundle in merged.bundles for sid in bundle.source_ids}
        merged.sources = {sid: source for sid, source in historic.sources.items() if sid in used}
    return merged


def citation_errors(report: str, catalog: SourceCatalog) -> list[str]:
    """Structural guard only: semantic support still requires the independent judge."""
    errors = []
    cited = set(_cited_ids(report))
    unknown = cited - catalog.sources.keys()
    if unknown:
        errors.append("Unknown source ids: " + ", ".join(sorted(unknown)))
    sections = re.split(r"(?im)^#{1,6}\s+Sources\s*:?\s*$", report, maxsplit=1)
    if len(sections) != 2:
        return [*errors, "Missing Sources section"]
    entries = {}
    for line in sections[1].splitlines():
        ids = _cited_ids(line)
        urls = re.findall(r"https?://[^\s<>\)\]]+", line)
        for sid in ids:
            entries.setdefault(sid, []).extend(urls)
    for sid in cited & catalog.sources.keys():
        urls = entries.get(sid, [])
        expected = _normalize_url(catalog.sources[sid].url)
        if not urls or any(_normalize_url(url) != expected for url in urls):
            errors.append(f"Missing or mismatched Sources URL for [{sid}]")
    return errors


def save_catalog_state(vfs, run_id: str, evidence_refs: list[dict]):
    """Record which evidence bundles the latest verification used."""
    return vfs.write_json(
        run_id,
        CATALOG_STATE_PATH,
        {"task_ids": evidence_task_ids(evidence_refs)},
        "verification_catalog",
    )


def _cited_ids(text: str) -> list[str]:
    """Every ``S<n>`` id cited in `text`, in order (``[S1, S2]`` counts both)."""
    return [sid for group in _CITATION.findall(text) for sid in re.findall(r"S\d+", group)]


def failure_fingerprint(result: VerificationResult) -> str:
    """Identity of *what* failed; an unchanged fingerprint after a fix means no progress."""
    if result.passed:
        return ""
    failed = sorted(
        (check.name, tuple(sorted(check.affected_ids)))
        for check in result.check_results
        if not check.passed
    )
    return hashlib.sha256(json.dumps(failed).encode()).hexdigest()
