"""Planner agent: splits the brief into research topics, and revises them on human feedback."""

from __future__ import annotations

from deep_research.prompts import PLAN_REVISION, PLANNER_PROMPT
from deep_research.schemas.research_plan import ResearchPlan
from deep_research.states.research_state import ResearchState
from deep_research.utils.text import bullets, today


class PlannerAgent:
    def __init__(self, llm):
        """Configure the structured model for producing research topics.

        Args:
            llm: Model tier used to create and revise research plans.
        """
        self.llm = llm.structured(ResearchPlan)

    def plan(self, state: ResearchState):
        """Create a research plan or revise the current topics using reviewer feedback.

        Args:
            state: Research brief and optional existing plan with revision feedback.

        Returns:
            Topic list and cleared plan feedback.
        """
        revision = ""
        if state.get("plan_feedback"):
            revision = PLAN_REVISION.format(plan=bullets(state["plan"]), feedback=state["plan_feedback"])
        prompt = PLANNER_PROMPT.format(date=today(), brief=state["brief"], revision=revision)
        return {"plan": self.llm.invoke(prompt).topics, "plan_feedback": ""}
