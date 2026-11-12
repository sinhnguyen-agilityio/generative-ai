"""Scope agent: asks up to three clarifying questions when needed, then writes the research brief and its
acceptance criteria. After a report, it decides what the user's follow-up message needs."""

from __future__ import annotations

from langchain_core.messages import AIMessage, get_buffer_string

from deep_research.prompts import BRIEF_PROMPT, CLARIFY_PROMPT, TRIAGE_PROMPT
from deep_research.schemas.clarify_with_user import ClarifyWithUser
from deep_research.schemas.follow_up import FollowUp
from deep_research.schemas.research_brief import ResearchBrief
from deep_research.states.research_state import ResearchState
from deep_research.utils.report import SOURCES_HEADING
from deep_research.utils.text import bullets, today


MAX_CLARIFY_QUESTIONS = 3


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
        asked = state.get("clarify_rounds", 0)
        prompt = CLARIFY_PROMPT.format(
            date=today(),
            messages=get_buffer_string(state["messages"]),
            asked=asked,
            max_questions=MAX_CLARIFY_QUESTIONS,
        )
        result = self.clarifier.invoke(prompt)
        ask = result.need_clarification and asked < MAX_CLARIFY_QUESTIONS
        reply = (
            result.question
            if ask
            else result.verification or "I have enough information. Starting the research."
        )
        return {"needs_clarification": ask, "clarify_rounds": asked + int(ask), "messages": [AIMessage(reply)]}

    def write_brief(self, state: ResearchState):
        """Convert the conversation into a research brief and acceptance criteria.

        Args:
            state: Conversation defining the research request.

        Returns:
            Brief text, nonempty stripped acceptance criteria, and a reset clarification budget for
            the next research request in this thread.
        """
        prompt = BRIEF_PROMPT.format(date=today(), messages=get_buffer_string(state["messages"]))
        result = self.brief_writer.invoke(prompt)
        return {
            "brief": result.brief,
            "criteria": [c.strip() for c in result.criteria if c.strip()],
            "clarify_rounds": 0,
        }

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
