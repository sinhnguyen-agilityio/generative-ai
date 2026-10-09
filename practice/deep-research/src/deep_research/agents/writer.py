"""Writer agent: writes the report from the findings, and rewrites it on human feedback or on the
problems the verifier found.

The findings carry one source numbering for the whole run (see ``merge_notes``); the model cites
those numbers and the Sources section is added here, listing only the sources the report cites."""

from __future__ import annotations

import re

from deep_research.prompts import REPORT_FIX, REPORT_PROMPT, REPORT_REVISION
from deep_research.states.research_state import ResearchState
from deep_research.utils.notes import merge_notes
from deep_research.utils.report import SOURCES_HEADING, with_sources
from deep_research.utils.text import CITATION, bullets, today

_OWN_SOURCES = re.compile(r"\n#+ *Sources\s*\n")  # a list the model wrote despite the prompt


class WriterAgent:
    def __init__(self, llm):
        """Configure the report-writing model without tool binding.

        Args:
            llm: Model tier used for drafting and revising reports.
        """
        self.llm = llm.plain()

    def write(self, state: ResearchState):
        """Write or revise the report from globally numbered research notes.

        Verification issues take precedence over reviewer feedback. Remove unknown citation
        numbers and replace any model-written source list with one built from trusted metadata.

        Args:
            state: Brief, notes, optional criteria, previous report, and revision feedback.

        Returns:
            Report with cited sources, cleared reviewer feedback, and reset evaluation.
        """
        notes, sources = merge_notes(state["notes"])
        previous = state.get("report", "").partition(SOURCES_HEADING)[0]
        issues = (state.get("verification") or {}).get("issues")
        revision = ""
        if issues:
            revision = REPORT_FIX.format(report=previous, issues=bullets(issues))
        elif state.get("report_feedback"):
            revision = REPORT_REVISION.format(report=previous, feedback=state["report_feedback"])
        prompt = REPORT_PROMPT.format(
            date=today(), brief=state["brief"], criteria=bullets(state.get("criteria") or []), notes=notes, revision=revision
        )
        text = _OWN_SOURCES.split(self.llm.invoke(prompt).text)[0].strip()
        text = CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in sources else "", text)
        return {"report": with_sources(text, sources), "report_feedback": "", "evaluation": None}
