"""Writer agent: writes the report from the findings, and rewrites it on human feedback."""

from __future__ import annotations

from deep_research.agents.utils import today
from deep_research.prompts import REPORT_PROMPT, REPORT_REVISION
from deep_research.state import ResearchState


class WriterAgent:
    def __init__(self, llm):
        self.llm = llm.plain()

    def write(self, state: ResearchState):
        revision = ""
        if state.get("report_feedback"):
            revision = REPORT_REVISION.format(report=state["report"], feedback=state["report_feedback"])
        prompt = REPORT_PROMPT.format(
            date=today(), brief=state["brief"], notes="\n\n---\n\n".join(state["notes"]), revision=revision
        )
        return {"report": self.llm.invoke(prompt).text, "report_feedback": "", "evaluation": None}
