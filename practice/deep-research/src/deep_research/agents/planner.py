"""Planner agent: splits the brief into research topics, and revises them on human feedback."""

from __future__ import annotations

from deep_research.agents.utils import bullets, today
from deep_research.prompts import PLAN_REVISION, PLANNER_PROMPT
from deep_research.state import ResearchPlan, ResearchState


class PlannerAgent:
    def __init__(self, llm):
        self.llm = llm.structured(ResearchPlan)

    def plan(self, state: ResearchState):
        revision = ""
        if state.get("plan_feedback"):
            revision = PLAN_REVISION.format(plan=bullets(state["plan"]), feedback=state["plan_feedback"])
        prompt = PLANNER_PROMPT.format(date=today(), brief=state["brief"], revision=revision)
        return {"plan": self.llm.invoke(prompt).topics, "plan_feedback": ""}
