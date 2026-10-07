"""Claim-level faithfulness gate (spec 002, FR-022 to FR-028; contracts/claim-evaluation.md).

Every sentence of the scored report body is one claim. The judge reads, for each sentence, only
the text the writer was shown for its cited sources (the draft's evidence snapshot) and says
whether the sentence states a fact, how far it is supported, and whether each cited source
supports part of it. Code then applies the rules: an uncited factual sentence, an unknown
source, a cited source that supports nothing, a partly supported, unsupported or contradicted
sentence all fail. The draft passes only when no factual sentence fails.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from intelligence_collector.agents.prompts import CLAIM_JUDGE_TEMPLATE
from intelligence_collector.contracts import (
    Claim,
    ClaimEvaluation,
    ClaimJudgeBatch,
    ClaimVerdict,
    VerificationCheck,
)
from intelligence_collector.observability.evaluation import report_sections
from intelligence_collector.observability.tracing import traced
from intelligence_collector.ports import ModelRouterPort

BATCH_SIZE = 10
CONCURRENCY = 4
MAX_FAILING_SENTENCES = 8
SOURCE_TEXT_CHARS = 3000
_CITATION = re.compile(r"\[(S\d+(?:\s*[,;]\s*S\d+)*)\]")
_HEADING = re.compile(r"^#{1,6}\s")
_TABLE_RULE = re.compile(r"^\|?\s*:?-{3,}")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
# A sentence ends at . ? ! (plus any citations right after it) followed by whitespace and an
# uppercase letter, digit, quote or bracket. "3.13" or "e.g. this" never split.
_SENTENCE_END = re.compile(
    r"(?<=[.?!])((?:\s*\[S\d+(?:\s*[,;]\s*S\d+)*\])*)\s+(?=[A-Z0-9\"'(\[])"
)


@dataclass(frozen=True)
class Sentence:
    claim_id: str
    section: str
    text: str
    cited: tuple[str, ...]


def cited_ids(text: str) -> list[str]:
    """Catalog ids cited in `text`, in order, without repeats."""
    ids: list[str] = []
    for group in _CITATION.findall(text):
        ids.extend(part.strip() for part in re.split(r"[,;]", group))
    return list(dict.fromkeys(ids))


def _split_paragraph(text: str) -> list[str]:
    pieces, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        end = match.start() + len(match.group(1))
        pieces.append(text[start:end])
        start = match.end()
    pieces.append(text[start:])
    return [piece.strip() for piece in pieces if piece.strip()]


def segment(report: str) -> list[Sentence]:
    """The scored sentences of `report`. ``report_sections`` already drops the Scope,
    Limitations and Sources sections, the title, gap disclosures and ``Assessment:`` sentences;
    here each list item and each table row with a citation is one sentence, headings are
    skipped and paragraphs are split into sentences."""
    found: list[tuple[str, str]] = []
    for section_text in report_sections(report):
        section, paragraph = "", []

        def flush() -> None:
            if paragraph:
                found.extend((section, piece) for piece in _split_paragraph(" ".join(paragraph)))
                paragraph.clear()

        for line in section_text.splitlines():
            stripped = line.strip()
            if not stripped:
                flush()
            elif _HEADING.match(stripped):
                flush()
                section = stripped.lstrip("#").strip()
            elif stripped.startswith("|"):
                flush()
                if _TABLE_RULE.match(stripped):
                    continue
                row = " | ".join(c.strip() for c in stripped.strip("|").split("|") if c.strip())
                if row and cited_ids(row):  # header rows carry no citations
                    found.append((section, row))
            elif _LIST_ITEM.match(stripped):
                flush()
                found.extend((section, piece) for piece in _split_paragraph(_LIST_ITEM.sub("", stripped)))
            else:
                paragraph.append(stripped)
        flush()
    return [
        Sentence(f"K{n}", section, text, tuple(cited_ids(text)))
        for n, (section, text) in enumerate(found, start=1)
    ]


def rule_failures(claim: Claim, known: set[str]) -> list[str]:
    """FR-024 and FR-026 applied to one judged sentence."""
    if not claim.states_fact:
        return []
    if not claim.cited_source_ids:
        return ["uncited"]
    failures = []
    if any(sid not in known for sid in claim.cited_source_ids):
        failures.append("unknown_source")
    judged = {item.get("source_id"): item.get("supports_part") for item in claim.per_source}
    nothing = [
        sid for sid in claim.cited_source_ids if sid in known and judged.get(sid) is not True
    ]
    if nothing:
        failures.append("source_supports_nothing:" + ",".join(nothing))
    if claim.verdict in {ClaimVerdict.PARTLY_SUPPORTED, ClaimVerdict.UNSUPPORTED}:
        failures.append("not_fully_supported")
    elif claim.verdict == ClaimVerdict.CONTRADICTED:
        failures.append("contradicted")
    return failures


class ClaimJudge:
    """The faithfulness gate (role ``claim_judge``, reasoning tier)."""

    def __init__(self, router: ModelRouterPort):
        self.router = router

    def _ask(self, batch: list[Sentence], snapshot: dict, run_id: str, thread_id: str):
        blocks = []
        for sentence in batch:
            sources = "\n".join(
                f"  [{sid}] {snapshot[sid].get('title', '')}: "
                f"{str(snapshot[sid].get('text', ''))[:SOURCE_TEXT_CHARS]}"
                for sid in sentence.cited
                if sid in snapshot
            ) or "  (no cited source text)"
            blocks.append(f"{sentence.claim_id}: {sentence.text}\n Cited sources:\n{sources}")
        return self.router.invoke_messages(
            "claim_judge",
            CLAIM_JUDGE_TEMPLATE.format_messages(claims="\n\n".join(blocks)),
            ClaimJudgeBatch,
            run_id=run_id,
            thread_id=thread_id,
        )

    async def _judge(self, sentences, snapshot, run_id, thread_id) -> dict:
        limit = asyncio.Semaphore(CONCURRENCY)
        batches = [sentences[i : i + BATCH_SIZE] for i in range(0, len(sentences), BATCH_SIZE)]

        async def one(batch):
            async with limit:
                return await asyncio.to_thread(self._ask, batch, snapshot, run_id, thread_id)

        found: dict = {}
        for result in await asyncio.gather(*(one(batch) for batch in batches)):
            wanted = {s.claim_id for s in sentences}
            for item in result.claims:
                if item.claim_id in wanted:
                    found.setdefault(item.claim_id, item)
        return found

    def evaluate(
        self,
        report: str,
        snapshot: dict,
        *,
        draft_version: int,
        artifact_hash: str,
        run_id: str = "",
        thread_id: str = "",
    ) -> ClaimEvaluation:
        """Judge every scored sentence against `snapshot` (source id -> title/url/text). Raises
        ``ValueError`` (an ERROR) when verdicts are still missing after one supplemental call."""
        sentences = segment(report)
        with traced(
            "claim_judge", as_type="evaluator", metadata={"sentences": len(sentences)}
        ) as observation:
            found = asyncio.run(self._judge(sentences, snapshot, run_id, thread_id)) if sentences else {}
            missing = [s for s in sentences if s.claim_id not in found]
            if missing:
                found.update(asyncio.run(self._judge(missing, snapshot, run_id, thread_id)))
            still = [s.claim_id for s in sentences if s.claim_id not in found]
            if still:
                raise ValueError("Claim judge returned no verdict for " + ", ".join(still))
            known = set(snapshot)
            claims = []
            for sentence in sentences:
                item = found[sentence.claim_id]
                claim = Claim(
                    claim_id=sentence.claim_id,
                    section=sentence.section,
                    sentence=sentence.text,
                    cited_source_ids=list(sentence.cited),
                    # A sentence that cites a source presents sourced content: always a claim.
                    states_fact=item.states_fact or bool(sentence.cited),
                    verdict=item.verdict,
                    per_source=[s.model_dump(exclude={"schema_version"}) for s in item.per_source],
                    unsupported_parts=list(item.unsupported_parts),
                )
                claim.failures = rule_failures(claim, known)
                claims.append(claim)
            factual = [c for c in claims if c.states_fact]
            supported = [c for c in factual if not c.failures]
            evaluation = ClaimEvaluation(
                draft_version=draft_version,
                artifact_hash=artifact_hash,
                claims=claims,
                supported_share=len(supported) / len(factual) if factual else 1.0,
                passed=len(supported) == len(factual),
            )
            observation.output = {"claims": len(claims), "failed": len(factual) - len(supported)}
        return evaluation


def claim_check(evaluation: ClaimEvaluation) -> VerificationCheck:
    """The gate result as a verification check; ``repair.diagnose`` routes it to ``vfs``."""
    failing = [c for c in evaluation.claims if c.failures]
    return VerificationCheck(
        name="claims",
        passed=evaluation.passed,
        gate="faithfulness",
        details=[]
        if evaluation.passed
        else [
            f"{c.claim_id}: {'; '.join(c.failures)} — {c.sentence[:200]}"
            + (f" (unsupported: {'; '.join(c.unsupported_parts)})" if c.unsupported_parts else "")
            for c in failing
        ],
        missing_points=[c.sentence for c in failing][:MAX_FAILING_SENTENCES],
        affected_ids=[f"claim:{c.claim_id}" for c in failing],
    )
