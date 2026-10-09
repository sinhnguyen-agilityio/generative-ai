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

from langchain_core.runnables import RunnableConfig

from deep_research.prompts import VERIFY_PROMPT
from deep_research.schemas.report_check import ReportCheck
from config.settings import Settings
from deep_research.states.research_state import ResearchState
from deep_research.utils import tracing
from deep_research.utils.notes import merge_notes, passages_by_url
from deep_research.utils.report import SOURCES_HEADING, with_sources
from deep_research.utils.runtime import thread_id
from deep_research.utils.verification import evidence, parse, remove

logger = logging.getLogger(__name__)


MAX_FIX_ROUNDS = 2


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
