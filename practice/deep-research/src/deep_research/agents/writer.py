"""Writer agent: writes the report from the findings, and rewrites it on human feedback.

The findings carry one source numbering for the whole run (see ``merge_notes``); the model cites
those numbers and the Sources section is added here, listing only the sources the report cites."""

from __future__ import annotations

import re

from deep_research.agents.utils import CITATION, merge_notes, today
from deep_research.prompts import REPORT_PROMPT, REPORT_REVISION
from deep_research.state import ResearchState

SOURCES_HEADING = "\n\n## Sources\n"
_OWN_SOURCES = re.compile(r"\n#+ *Sources\s*\n")  # a list the model wrote despite the prompt


class WriterAgent:
    def __init__(self, llm):
        self.llm = llm.plain()

    def write(self, state: ResearchState):
        notes, sources = merge_notes(state["notes"])
        revision = ""
        if state.get("report_feedback"):
            previous = state["report"].partition(SOURCES_HEADING)[0]
            revision = REPORT_REVISION.format(report=previous, feedback=state["report_feedback"])
        prompt = REPORT_PROMPT.format(date=today(), brief=state["brief"], notes=notes, revision=revision)
        text = _OWN_SOURCES.split(self.llm.invoke(prompt).text)[0].strip()
        text = CITATION.sub(lambda m: m.group(0) if int(m.group(1)) in sources else "", text)
        cited = sorted({int(n) for n in CITATION.findall(text)})
        if cited:
            text += SOURCES_HEADING + "\n".join(f"[{n}] {sources[n][0]}: {sources[n][1]}" for n in cited)
        return {"report": text, "report_feedback": "", "evaluation": None}
