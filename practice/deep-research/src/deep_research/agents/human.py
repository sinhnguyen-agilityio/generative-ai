"""Human agent: pauses the graph (LangGraph interrupt) for the reviewer's decision.

The UI resumes with ``{"approved": True, "topics": [...]}`` or ``{"approved": False, "feedback": "..."}``.
Feedback left in state sends the run back for revision; empty feedback means approved.
"""

from __future__ import annotations

from langgraph.types import interrupt

from deep_research.state import ResearchState


def _feedback(decision: dict, default: str) -> str:
    return "" if decision.get("approved") else decision.get("feedback") or default


class HumanAgent:
    def review_plan(self, state: ResearchState):
        decision = interrupt({"type": "plan", "topics": state["plan"]})
        feedback = _feedback(decision, "Please revise the plan.")
        if feedback:
            return {"plan_feedback": feedback}
        topics = [t.strip() for t in decision.get("topics") or state["plan"] if t.strip()]
        return {"plan": topics, "plan_feedback": ""}

    def review_report(self, state: ResearchState):
        decision = interrupt({"type": "report", "report": state["report"], "evaluation": state.get("evaluation")})
        return {"report_feedback": _feedback(decision, "Please revise the report.")}
