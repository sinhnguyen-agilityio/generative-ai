"""Programmatic source-verification script for research reports (the "Verify" in BVF).

Deterministic checks of the report text against the run's stored evidence bundles; no LLM is
involved, so a report cannot talk its way past verification. Run it standalone with::

    python -m intelligence_collector.source_verification --run-id RUN [--vfs-root .data/vfs]

It prints the ``VerificationResult`` JSON and exits 0 only when every check passes.

Report contract checked here (the report writer's prompt states the same rules):

* one ``# `` title, at least one body section, and a final ``## Sources`` section;
* every sentence outside ``## Scope`` / ``## Limitations`` / ``## Sources`` cites ``[S<n>]``;
* cited ids exist, and the cited source text supports the claim (numbers must appear verbatim,
  wording must overlap);
* no URL that is not in the evidence; ``## Sources`` lists every cited id with its exact URL;
* every researched question is cited at least once;
* unresolved evidence gaps are researched once, then disclosed under ``## Limitations``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from intelligence_collector.contracts import IssueType, VerificationCheck, VerificationResult
from intelligence_collector.tools.url_tracking import normalize_url

CATALOG_STATE_PATH = "verification/catalog_state.json"
EXEMPT_SECTIONS = ("scope", "limitations", "sources", "references")
MIN_CLAIM_WORDS = 4
MIN_WORD_OVERLAP = 0.35
MIN_GAP_DISCLOSURE_OVERLAP = 0.5

_CITATION = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")
_URL = re.compile(r"https?://[^\s)\]>\"']+")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_ABBREV_WORD = re.compile(r"([A-Za-z]+)\.$")
# Abbreviations that are frequently followed by a numeral (a day-of-month, a volume number, ...)
# without ending the sentence — e.g. "released on Aug. 4, 2026 [S1]." is one claim, not two.
_ABBREVIATIONS = frozenset(
    "jan feb mar apr jun jul aug sep sept oct nov dec no vol pp".split()
)
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
# "as of <today's date>" is the report's own generation-time framing, never something a source
# can state; stripped before grounding so it doesn't demand today's date appear in evidence text.
# Matches "as of 2026-09-28" or "as of September 28, 2026" / "as of Sept. 28, 2026".
_AS_OF_CLAUSE = re.compile(
    r"\bas of\b\s+(?:\d{4}-\d{2}-\d{2}|[A-Za-z]+\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4})",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    "about above after again also among another based because been before being between both "
    "cannot could does doing during each either every from further have having here into itself "
    "just made make many more most much must only other over same should some such than that "
    "their them then there these they this those through under until very were what when where "
    "which while will with within without would your".split()
)


# ===== EVIDENCE CATALOG =====


@dataclass(frozen=True)
class CatalogSource:
    source_id: str  # "S1", "S2", ... stable per unique URL
    url: str
    title: str
    excerpt: str


@dataclass(frozen=True)
class EvidenceGap:
    gap_id: str
    todo_id: str
    question: str
    text: str


@dataclass
class EvidenceBundle:
    todo_id: str
    question: str
    summary: str
    source_ids: list[str]
    gaps: list[EvidenceGap]


@dataclass
class SourceCatalog:
    sources: dict[str, CatalogSource] = field(default_factory=dict)
    bundles: list[EvidenceBundle] = field(default_factory=list)

    @property
    def gaps(self) -> list[EvidenceGap]:
        return [gap for bundle in self.bundles for gap in bundle.gaps]

    @property
    def urls(self) -> set[str]:
        return {_normalize_url(source.url) for source in self.sources.values()}


def _normalize_url(url: str) -> str:
    # URLs pulled out of report prose may carry trailing punctuation.
    return normalize_url(url.strip().rstrip(".,;:"))


def gap_id(todo_id: str, text: str) -> str:
    return f"gap:{todo_id}:{hashlib.sha1(text.strip().casefold().encode()).hexdigest()[:10]}"


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
            EvidenceGap(gap_id(todo_id, text), todo_id,
                        question, str(text).strip())
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
    return list(dict.fromkeys(ref["task_id"] for ref in evidence_refs if ref.get("task_id")))


def load_catalog(vfs, run_id: str, evidence_refs: list[dict]) -> SourceCatalog:
    """Read the evidence bundles referenced by graph state (in research order)."""
    bundles = [
        json.loads(
            vfs.read(run_id, f"research/{task_id}/evidence/result.json"))
        for task_id in evidence_task_ids(evidence_refs)
    ]
    return catalog_from_bundles(bundles)


def save_catalog_state(vfs, run_id: str, evidence_refs: list[dict], researched_gaps: list[str]):
    """Persist what the standalone script needs to reproduce the graph's verification."""
    return vfs.write_json(
        run_id,
        CATALOG_STATE_PATH,
        {"task_ids": evidence_task_ids(
            evidence_refs), "researched_gaps": list(researched_gaps)},
        "verification_catalog",
    )


# ===== REPORT PARSING =====


def _sections(report: str) -> list[tuple[str, list[str]]]:
    sections: list[tuple[str, list[str]]] = [("", [])]
    for line in report.splitlines():
        heading = re.match(r"^##\s+(.+?)\s*#*\s*$", line)
        if heading:
            sections.append((heading.group(1).strip().casefold(), []))
        else:
            sections[-1][1].append(line)
    return sections


def _is_exempt(title: str) -> bool:
    return any(title.startswith(name) for name in EXEMPT_SECTIONS)


def _split_sentences(line: str) -> list[str]:
    """Split on sentence-ending punctuation, but not after an abbreviation like "Aug." or
    "Sept." — those are followed by a day number, not a new sentence, and splitting there would
    strand the citation (attached to the date that follows) away from the claim it supports."""
    sentences = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(line):
        before = line[start:match.start()]
        word = _ABBREV_WORD.search(before)
        if word and word.group(1).casefold() in _ABBREVIATIONS:
            continue
        sentences.append(before)
        start = match.end()
    sentences.append(line[start:])
    return sentences


def _cited_ids(text: str) -> list[str]:
    return [sid for group in _CITATION.findall(text) for sid in re.findall(r"S\d+", group)]


def _strip_markup(text: str) -> str:
    return _URL.sub(" ", _CITATION.sub(" ", text))


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    cited: tuple[str, ...]


def extract_claims(report: str) -> list[Claim]:
    """Every sentence (or table row) of the body sections is a claim that needs a citation."""
    claims = []
    for title, lines in _sections(report):
        if _is_exempt(title):
            continue
        in_code = False
        for raw in lines:
            line = raw.strip()
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code or not line or line.startswith("#") or re.fullmatch(r"[|:\-\s]+", line):
                continue
            line = _LIST_MARKER.sub("", line).strip("| ").replace(" | ", " ")
            for sentence in _split_sentences(line):
                words = _strip_markup(sentence).split()
                if len(words) < MIN_CLAIM_WORDS:
                    continue
                digest = hashlib.sha1(
                    " ".join(words).casefold().encode()).hexdigest()[:10]
                claims.append(
                    Claim(f"claim:{digest}", sentence.strip(), tuple(_cited_ids(sentence))))
    return claims


def _stem(word: str) -> str:
    for suffix in ("ingly", "edly", "ing", "ed", "es", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _tokens(text: str) -> tuple[set[str], set[str]]:
    """Return (numbers, significant word stems) of a text."""
    numbers, words = set(), set()
    for token in _TOKEN.findall(_strip_markup(text).casefold()):
        if any(ch.isdigit() for ch in token):
            numbers.add(token.replace(",", ""))
        elif len(token) >= 4 and token not in _STOPWORDS:
            words.add(_stem(token))
    return numbers, words


def _number_supported(number: str, source_numbers: set[str]) -> bool:
    return any(n == number or n.startswith(number + ".") for n in source_numbers)


def _short(text: str, limit: int = 140) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ===== CHECKS =====


def _check(name, failures, issue=IssueType.DRAFT_DEFECT) -> VerificationCheck:
    details = [detail for detail, _ in failures]
    affected = list(dict.fromkeys(affected for _, affected in failures))
    return VerificationCheck(
        name=name,
        passed=not failures,
        issue_type=None if not failures else issue,
        details=details,
        affected_ids=affected,
    )


def verify_report(
    report: str,
    catalog: SourceCatalog,
    *,
    researched_gaps: list[str] | tuple[str, ...] = (),
) -> VerificationResult:
    """Run every source-verification check; the report passes only if all of them pass."""
    # Hash the exact report text now -- this hash is what the human-approval step later binds
    # its decision to, so it must be computed before any check can mutate/replace `report`.
    artifact_hash = hashlib.sha256(
        report.encode()).hexdigest() if report else ""
    # Every check below appends one VerificationCheck to this list; the report only passes if
    # none of them recorded a failure.
    checks: list[VerificationCheck] = []

    # Check 1: evidence_present -- the run must have at least one stored source to cite at all.
    checks.append(_check("evidence_present", [] if catalog.sources else [
        ("The run has no stored evidence sources to cite.", "evidence")
    ], IssueType.EVIDENCE_GAP))
    # `researched_gaps` is the state's record of which evidence gaps already got a follow-up
    # search; turned into a set for O(1) membership checks below.
    researched = set(researched_gaps)
    # Check 2: evidence_gaps_researched -- every gap the research phase flagged must have been
    # looked into once (via researching_gaps) before the report can be verified; one failure
    # entry per gap that was never followed up on.
    checks.append(_check("evidence_gaps_researched", [
        (f"Unresearched gap for '{_short(gap.question, 80)}': {gap.text}", gap.gap_id)
        for gap in catalog.gaps
        if gap.gap_id not in researched
    ], IssueType.EVIDENCE_GAP))

    # Parse the report into (heading, body-lines) sections once; every later check reuses this.
    sections = _sections(report)
    # Section titles excluding the "" preamble section that comes before the first heading.
    titles = [title for title, _ in sections[1:]]
    # First non-blank line of the report, used to check for the required "# Title".
    first_line = next((line.strip()
                      for line in report.splitlines() if line.strip()), "")
    structure = []
    # The report must open with a Markdown H1 title.
    if not first_line.startswith("# "):
        structure.append(
            ("The report must start with a single '# ' title line.", "structure:title"))
    # The report must end with a "## Sources" (or "## References") section.
    if not any(title.startswith(("sources", "references")) for title in titles):
        structure.append(
            ("The report must end with a '## Sources' section.", "structure:sources"))
    # At least one real content section must exist -- Scope/Limitations/Sources alone isn't a
    # report.
    if not any(not _is_exempt(title) for title in titles):
        structure.append(
            ("The report needs at least one '## ' body section with findings.", "structure:body"))
    # Check 3: structure -- fails if any of the three structural rules above was violated.
    checks.append(_check("structure", structure))

    # Every sentence outside the exempt sections, each tagged with which [S<n>] ids it cites.
    claims = extract_claims(report)
    # Every citation id mentioned anywhere in the whole report (used just below).
    all_cited = _cited_ids(report)
    # Check 4: citation_ids_known -- every cited id must actually exist in the evidence catalog
    # (guards against a hallucinated source number).
    checks.append(_check("citation_ids_known", [
        (f"Citation [{sid}] is not in the source catalog.", f"citation:{sid}")
        for sid in dict.fromkeys(all_cited)
        if sid not in catalog.sources
    ]))
    # Check 5: claims_cited -- every claim sentence must cite at least one source.
    checks.append(_check("claims_cited", [
        (f"Uncited claim: \"{_short(claim.text)}\"", claim.claim_id)
        for claim in claims
        if not claim.cited
    ]))

    # Check 6: claims_grounded -- for every cited claim, the cited source text must actually
    # support it (not just exist).
    grounding = []
    for claim in claims:
        # The actual CatalogSource objects this claim cites (skips any unknown id, already
        # reported by citation_ids_known above).
        cited = [catalog.sources[sid]
                 for sid in claim.cited if sid in catalog.sources]
        if not cited:
            continue
        # Numbers and significant word-stems found in the combined title+excerpt of every
        # source this claim cites.
        source_numbers, source_words = _tokens(
            " ".join(f"{s.title} {s.excerpt}" for s in cited))
        # Same tokenization of the claim itself, with "as of <date>" stripped first since that
        # clause describes the report's own generation time, not something a source can state.
        numbers, words = _tokens(_AS_OF_CLAUSE.sub(" ", claim.text))
        # Human-readable "[S1][S2]"-style label for the error message below.
        label = "".join(f"[{s.source_id}]" for s in cited)
        # Any number in the claim (a stat, a date, a count) that doesn't appear verbatim (or as
        # a prefix, e.g. "12" matching "12.5") in the cited source text.
        missing = sorted(
            n for n in numbers if not _number_supported(n, source_numbers))
        if missing:
            # A missing number is grounds for failure on its own, regardless of word overlap.
            grounding.append((
                f"Number(s) {', '.join(missing)} in \"{_short(claim.text)}\" do not appear in {label}.",
                claim.claim_id,
            ))
        elif len(words) >= 3:
            # No numeric mismatch: fall back to word-overlap between the claim and its cited
            # source text -- too little overlap means the citation likely doesn't actually
            # support this specific sentence.
            overlap = len(words & source_words) / len(words)
            if overlap < MIN_WORD_OVERLAP:
                grounding.append((
                    f"\"{_short(claim.text)}\" is not supported by {label} "
                    f"(only {overlap:.0%} of its key terms appear in the cited source text).",
                    claim.claim_id,
                ))
    checks.append(_check("claims_grounded", grounding))

    # Every unique, normalized URL that exists anywhere in the evidence catalog.
    known_urls = catalog.urls
    # Check 7: urls_known -- every URL that appears in the report text must be one that was
    # actually retrieved as evidence (guards against a fabricated link).
    checks.append(_check("urls_known", [
        (f"URL {url} is not in the stored evidence.", f"url:{url}")
        for url in dict.fromkeys(_normalize_url(u) for u in _URL.findall(report))
        if url not in known_urls
    ]))

    # Map of source id -> URL, as actually written in the report's own "## Sources" section
    # (not the catalog -- this checks what the report *claims*, against what the catalog says).
    listed: dict[str, str] = {}
    for title, lines in sections:
        if title.startswith(("sources", "references")):
            for line in lines:
                ids, urls = _cited_ids(line), _URL.findall(line)
                if ids and urls:
                    listed[ids[0]] = _normalize_url(urls[0])
    # Every source id cited anywhere in the report body (used to cross-check against `listed`).
    body_cited = {sid for claim in claims for sid in claim.cited}
    # Every id cited in the body but missing from the "## Sources" listing.
    source_listing = [
        (f"[{sid}] is cited but not listed in '## Sources' with its URL.",
         f"sources:{sid}")
        for sid in sorted(body_cited & catalog.sources.keys(), key=lambda s: int(s[1:]))
        if sid not in listed
    ]
    # Every id that IS listed in "## Sources" but with the wrong URL (should match the catalog).
    source_listing += [
        (f"'## Sources' lists [{sid}] with {url}, but its URL is {catalog.sources[sid].url}.",
         f"sources:{sid}")
        for sid, url in listed.items()
        if sid in catalog.sources and url != _normalize_url(catalog.sources[sid].url)
    ]
    # Check 8: sources_section -- combines both kinds of listing mismatch above.
    checks.append(_check("sources_section", source_listing))

    # Check 9: question_coverage -- every research question that produced sources must have at
    # least one of those sources actually cited somewhere in the report body (otherwise the
    # research was wasted and the question goes unanswered in the final text).
    checks.append(_check("question_coverage", [
        (f"No claim cites evidence for the research question: {_short(bundle.question)}",
         f"question:{bundle.todo_id}")
        for bundle in catalog.bundles
        if bundle.source_ids and not set(bundle.source_ids) & body_cited
    ]))

    # Full text of the "## Limitations" section (if any), used to check gap disclosure below.
    limitations = " ".join(
        " ".join(lines) for title, lines in sections if title.startswith("limitations")
    )
    _, limitation_words = _tokens(limitations)
    undisclosed = []
    for gap in catalog.gaps:
        _, gap_words = _tokens(gap.text)
        # Fraction of the gap's own significant words that also appear under Limitations; an
        # empty gap counts as fully "overlapping" (nothing to disclose) rather than failing.
        overlap = len(gap_words & limitation_words) / \
            len(gap_words) if gap_words else 1.0
        if not limitations or overlap < MIN_GAP_DISCLOSURE_OVERLAP:
            undisclosed.append((
                f"Unresolved gap is not disclosed under '## Limitations': {gap.text}", gap.gap_id
            ))
    # Check 10: gaps_disclosed -- any gap that was never researched (see check 2) must at least
    # be honestly written up under Limitations rather than silently omitted.
    checks.append(_check("gaps_disclosed", undisclosed))

    # Collect every check that recorded at least one failure.
    failed = [check for check in checks if not check.passed]
    issue = None
    if failed:
        # One overall issue type for the whole result: EVIDENCE_GAP takes priority (it needs
        # more research, not just a rewrite) whenever any failed check was evidence-related;
        # otherwise it's a DRAFT_DEFECT (a wording/citation problem the writer can fix directly).
        issue = (
            IssueType.EVIDENCE_GAP
            if any(check.issue_type == IssueType.EVIDENCE_GAP for check in failed)
            else IssueType.DRAFT_DEFECT
        )
    return VerificationResult(
        artifact_hash=artifact_hash,
        # Passes only if literally zero checks failed.
        passed=not failed,
        checks=[check.name for check in checks],
        issue_type=issue,
        # Every affected id from every failed check, in order, with duplicates removed --
        # feeds the Fix step so it knows exactly what to target.
        affected_ids=list(dict.fromkeys(
            a for check in failed for a in check.affected_ids)),
        # Every failure detail message from every failed check, one per line, prefixed with
        # which check it came from -- this is what gets shown to the reviewer/writer.
        repair_details="\n".join(
            f"[{check.name}] {detail}" for check in failed for detail in check.details
        ),
        # Full per-check detail (not just names), used by callers that need more than the
        # summary, e.g. the Streamlit UI's failed-checks list.
        check_results=checks,
    )


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


# ===== COMMAND-LINE SCRIPT =====


def main(argv: list[str] | None = None) -> int:
    from intelligence_collector.config import Settings

    parser = argparse.ArgumentParser(
        description="Verify a research report against its sources.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--vfs-root", default=Settings().vfs_root)
    parser.add_argument(
        "--report", default="reports/latest.md", help="path inside the run")
    args = parser.parse_args(argv)

    run_dir = Path(args.vfs_root) / args.run_id
    # The graph records the catalog order so S<n> ids match the ones the report cites.
    state_path = run_dir / CATALOG_STATE_PATH
    state = json.loads(state_path.read_text(encoding="utf-8")
                       ) if state_path.exists() else {}
    task_ids = state.get("task_ids") or sorted(
        path.parent.parent.name for path in run_dir.glob("research/*/evidence/result.json")
    )
    bundles = [
        json.loads((run_dir / "research" / task_id / "evidence" /
                   "result.json").read_text(encoding="utf-8"))
        for task_id in task_ids
    ]
    researched = state.get("researched_gaps", [])
    report = (run_dir / args.report).read_text(encoding="utf-8")
    result = verify_report(report, catalog_from_bundles(
        bundles), researched_gaps=researched)
    print(result.model_dump_json(indent=2))
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
