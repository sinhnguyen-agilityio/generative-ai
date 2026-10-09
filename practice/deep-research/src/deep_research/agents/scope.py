"""Scope agent: asks one clarifying question when needed, then writes the research brief and its
acceptance criteria. After a report, it decides what the user's follow-up message needs."""

from __future__ import annotations

from langchain_core.messages import AIMessage, get_buffer_string

from deep_research.agents.utils import bullets, today
from deep_research.agents.writer import SOURCES_HEADING
from deep_research.prompts import BRIEF_PROMPT, CLARIFY_PROMPT, TRIAGE_PROMPT
from deep_research.state import ClarifyWithUser, FollowUp, ResearchBrief, ResearchState


class ScopeAgent:
    def __init__(self, llm):
        """Configure structured models for clarification, brief writing, and follow-up triage.

        Args:
            llm: Model tier used for research scoping.
        """
        self.clarifier = llm.structured(ClarifyWithUser)
        self.brief_writer = llm.structured(ResearchBrief)
        self.triager = llm.structured(FollowUp)

    def clarify(self, state: ResearchState):
        """Decide whether the conversation needs clarification before research begins.

        Args:
            state: Conversation messages containing the request and any prior clarification.

        Returns:
            Clarification flag and an assistant question or confirmation message.
        """
        prompt = CLARIFY_PROMPT.format(date=today(), messages=get_buffer_string(state["messages"]))
        result = self.clarifier.invoke(prompt)
        reply = result.question if result.need_clarification else result.verification
        return {"needs_clarification": result.need_clarification, "messages": [AIMessage(reply)]}

    def write_brief(self, state: ResearchState):
        """Convert the conversation into a research brief and acceptance criteria.

        Args:
            state: Conversation defining the research request.

        Returns:
            Brief text and nonempty, stripped acceptance criteria.
        """
        prompt = BRIEF_PROMPT.format(date=today(), messages=get_buffer_string(state["messages"]))
        result = self.brief_writer.invoke(prompt)
        return {"brief": result.brief, "criteria": [c.strip() for c in result.criteria if c.strip()]}

    def triage(self, state: ResearchState):
        """Classify report feedback as new research or a rewrite using existing evidence.

        Args:
            state: Brief, approved plan, current report, and reviewer feedback.

        Returns:
            Research request and reset supervisor counter when facts are needed; otherwise an empty request.
        """
        prompt = TRIAGE_PROMPT.format(
            brief=state["brief"],
            plan=bullets(state["plan"]),
            report=state["report"].partition(SOURCES_HEADING)[0],
            request=state["report_feedback"],
        )
        if self.triager.invoke(prompt).kind == "research":
            return {"research_request": state["report_feedback"], "supervisor_iterations": 0}
        return {"research_request": ""}
