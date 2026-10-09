"""Human agent: pauses the graph (LangGraph interrupt) for the user's decision.

The UI resumes the plan review with ``{"approved": True, "topics": [...], "criteria": [...]}`` or
``{"approved": False, "feedback": "..."}``. Feedback left in state sends the plan back to the planner.

The verified report is shown as the answer, like a chat reply. The run waits there: the user either
publishes it (``{"approved": True}``), the only permanent write, or sends a follow-up message
(``{"approved": False, "feedback": "..."}``). The scope agent's triage then decides whether the
follow-up needs more research or only a rewrite.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from deep_research.state import ResearchState


def _feedback(decision: dict, default: str) -> str:
    """Resolve a human decision into revision feedback.

    Args:
        decision: Review response with approval and optional feedback.
        default: Revision request used when approval and feedback are both absent.

    Returns:
        Empty text for approval; otherwise supplied feedback or the default.
    """
    return "" if decision.get("approved") else decision.get("feedback") or default


def _lines(items: list[str]) -> list[str]:
    """Strip review items and discard blank entries.

    Args:
        items: Edited topics or acceptance criteria.

    Returns:
        Nonempty items in their original order.
    """
    return [item.strip() for item in items if item.strip()]


class HumanAgent:
    def review_plan(self, state: ResearchState):
        """Pause for plan review and apply approval, edits, or revision feedback.

        Args:
            state: Proposed topic plan and optional acceptance criteria.

        Returns:
            Plan feedback for a revision, or approved topics and criteria with feedback cleared.
        """
        criteria = state.get("criteria") or []
        decision = interrupt({"type": "plan", "topics": state["plan"], "criteria": criteria})
        feedback = _feedback(decision, "Please revise the plan.")
        if feedback:
            return {"plan_feedback": feedback}
        return {
            "plan": _lines(decision.get("topics") or state["plan"]),
            "criteria": _lines(decision.get("criteria", criteria)),
            "plan_feedback": "",
        }

    def review_report(self, state: ResearchState):
        """Pause for report review and accept publication or a follow-up request.

        Args:
            state: Report and optional verification/evaluation summaries.

        Returns:
            Cleared feedback on approval; otherwise feedback, a reset fix budget, and chat messages.
        """
        decision = interrupt(
            {
                "type": "report",
                "report": state["report"],
                "verification": state.get("verification"),
                "evaluation": state.get("evaluation"),
            }
        )
        feedback = _feedback(decision, "Please revise the report.")
        if not feedback:
            return {"report_feedback": ""}
        # The answered report and the follow-up stay in the conversation, as in a chat.
        return {
            "report_feedback": feedback,
            "fix_rounds": 0,
            "messages": [AIMessage(state["report"]), HumanMessage(feedback)],
        }
