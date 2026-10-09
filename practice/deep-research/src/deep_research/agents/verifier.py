"""Verifier agent: checks the written report against the evidence of its sources and the brief's
acceptance criteria, and sends it back to the writer until it passes.

    write_report → verify ⇄ write_report (at most MAX_FIX_ROUNDS times) → evaluate

One structured call on the reasoning tier judges every sentence against the passages of the sources
it cites (not all passages: a claim attributed to the wrong source fails) and every criterion.
Code decides the rest: any unsupported sentence or unmet criterion sends the report back with the
problems listed; after MAX_FIX_ROUNDS fixes the unsupported sentences left are deleted, so every
report that reaches the reviewer passes the sentence check. Criteria still unmet are shown to the
reviewer, who can ask for more research. A failed check is recorded, not raised."""

from __future__ import annotations

import logging
import re

from langchain_core.runnables import RunnableConfig

from deep_research import tracing
from deep_research.agents.utils import CITATION, SENTENCE_END, merge_notes, passages_by_url, thread_id
from deep_research.agents.writer import SOURCES_HEADING, with_sources
from deep_research.config import Settings
from deep_research.prompts import VERIFY_PROMPT
from deep_research.state import ReportCheck, ResearchState

logger = logging.getLogger(__name__)

MAX_FIX_ROUNDS = 2
_TABLE_RULE = re.compile(r"^\|?[\s:|-]+\|?$")
_LIST_MARK = re.compile(r"^\s*(?:(?:[-*+]|\d+[.)])\s+|>\s*)?")


class VerifierAgent:
    def __init__(self, llm, settings: Settings):
        """Configure the structured report judge and optional tracing.

        Args:
            llm: Model tier producing sentence and criterion verdicts.
            settings: Runtime settings controlling Langfuse tracing.
        """
        self.judge = llm.structured(ReportCheck)
        self.langfuse = settings.langfuse_enabled

    def verify(self, state: ResearchState, config: RunnableConfig):
        """Check cited claims and acceptance criteria, requesting up to two writer corrections.

        After the fix budget is exhausted, remove sentences judged unsupported and rebuild
        the source list. Judge failures are recorded in state rather than raised.

        Args:
            state: Brief, report, source notes, acceptance criteria, and current fix count.
            config: Graph configuration used to attach verification metrics to the thread.

        Returns:
            Verification issues and incremented fix count, a final report/verification summary, or a recorded error.
        """
        lines = parse(state["report"].partition(SOURCES_HEADING)[0])
        units = [(i, j, s) for i, (_, sentences) in enumerate(lines) if sentences for j, s in enumerate(sentences)]
        criteria = state.get("criteria") or []
        _, sources = merge_notes(state["notes"])
        try:
            check = self.judge.invoke(
                VERIFY_PROMPT.format(
                    brief=state["brief"],
                    criteria="\n".join(f"C{n}: {c}" for n, c in enumerate(criteria, 1)) or "None.",
                    sentences="\n".join(f"S{n}: {s}" for n, (_, _, s) in enumerate(units, 1)),
                    evidence=evidence(units, sources, passages_by_url(state["notes"])),
                )
            )
        except Exception as exc:
            logger.exception("Verification failed")
            return {"verification": {"error": f"{type(exc).__name__}: {exc}"}}

        unsupported = {v.sentence: v.problem for v in check.sentences if v.verdict == "unsupported" and 0 < v.sentence <= len(units)}
        unmet = {c.criterion: c.problem for c in check.criteria if not c.met and 0 < c.criterion <= len(criteria)}
        claims = sum(v.verdict != "no_claim" for v in check.sentences if 0 < v.sentence <= len(units))
        support = 1 - len(unsupported) / claims if claims else 1.0
        rounds = state.get("fix_rounds", 0)
        tracing.score(self.langfuse, thread_id(config), "citation_support", support, f"draft {rounds + 1}")

        if (unsupported or unmet) and rounds < MAX_FIX_ROUNDS:
            issues = [f'S{n} "{units[n - 1][2]}": {problem}' for n, problem in sorted(unsupported.items())]
            issues += [f"Criterion not answered: {criteria[n - 1]} ({problem})" for n, problem in sorted(unmet.items())]
            return {"verification": {"issues": issues}, "fix_rounds": rounds + 1}

        report = state["report"]
        removed = [units[n - 1][2] for n in sorted(unsupported)]
        if removed:
            body = remove(lines, {units[n - 1][:2] for n in unsupported})
            report = with_sources(body, sources)
        tracing.score(self.langfuse, thread_id(config), "sentences_removed", len(removed))
        verification = {"support": support, "rounds": rounds, "removed": removed, "unmet": [criteria[n - 1] for n in sorted(unmet)]}
        return {"report": report, "verification": verification}


def parse(body: str) -> list[tuple[str, list[str] | None]]:
    """Split a Markdown body into verifiable sentences while preserving structural lines.

    Headings, blank lines, table rules, and fenced code have no sentences. Other lines
    keep their list marker separately; a table row is treated as one sentence.

    Args:
        body: Report body without its generated Sources section.

    Returns:
        Pairs of preserved text/list marker and a sentence list, or None for structural lines.
    """
    lines, fenced = [], False
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
        if fenced or not stripped or stripped.startswith(("```", "#")) or _TABLE_RULE.match(stripped):
            lines.append((line, None))
        elif stripped.startswith("|"):
            lines.append(("", [line]))
        else:
            mark = _LIST_MARK.match(line).group(0)
            lines.append((mark, SENTENCE_END.split(line[len(mark):].strip())))
    return lines


def remove(lines: list[tuple[str, list[str] | None]], drop: set[tuple[int, int]]) -> str:
    """Remove selected sentences, empty content lines, and sections left without content.

    Args:
        lines: Parsed report lines from parse.
        drop: Zero-based (line index, sentence index) pairs to remove.

    Returns:
        Rebuilt Markdown body with redundant blank lines collapsed.
    """
    kept = []
    for i, (text, sentences) in enumerate(lines):
        if sentences is None:
            kept.append(text)
        elif rest := [s for j, s in enumerate(sentences) if (i, j) not in drop]:
            kept.append(text + " ".join(rest))
    body, after = [], None  # heading level of the next non-blank line kept; 0 for content
    for line in reversed(kept):
        level = len(line) - len(line.lstrip("#"))
        if level > 1 and (after is None or 0 < after <= level):
            continue
        body.append(line)
        if line.strip():
            after = level
    return re.sub(r"\n{3,}", "\n\n", "\n".join(reversed(body))).strip()


def evidence(units, sources: dict[int, tuple[str, str]], passages: dict[str, list[str]]) -> str:
    """Format evidence for the known sources cited by the report sentences.

    Args:
        units: Indexed sentence tuples (line index, sentence index, text).
        sources: Global source numbers mapped to (title, URL).
        passages: Read source passages grouped by URL.

    Returns:
        Source-numbered passages with placeholders for unread sources, or "None.".
    """
    cited = sorted({int(n) for _, _, s in units for n in CITATION.findall(s)} & sources.keys())
    blocks = []
    for n in cited:
        title, url = sources[n]
        texts = passages.get(url) or ["(no passage of this source was read)"]
        blocks.append(f"[{n}] {title}\n" + "\n---\n".join(texts))
    return "\n\n".join(blocks) or "None."
