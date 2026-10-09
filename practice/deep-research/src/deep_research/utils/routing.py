"""Select the next node in the main research workflow."""

from __future__ import annotations

from langgraph.graph import END

from deep_research.states.research_state import ResearchState


def next_after_clarify(state: ResearchState):
    """End the turn when clarification is needed; otherwise write the brief.

    Args:
        state: Clarification decision from the scope agent.

    Returns:
        END or the write_brief node name.
    """
    if state["needs_clarification"]:
        return END
    return "write_brief"


def next_after_guardrail(state: ResearchState):
    """End blocked requests; otherwise start clarification.

    Args:
        state: Guardrail decision for the current request.

    Returns:
        END or the clarify node name.
    """
    if state["blocked"]:
        return END
    return "clarify"


def next_after_review_plan(state: ResearchState):
    """Route requested plan revisions back to the planner.

    Args:
        state: Human plan feedback after review.

    Returns:
        planner when feedback exists; otherwise supervisor.
    """
    if state["plan_feedback"]:
        return "planner"
    return "supervisor"


def next_after_verify(state: ResearchState):
    """Route verification issues back to the writer for correction.

    Args:
        state: Verification result for the current report.

    Returns:
        write_report when issues exist; otherwise evaluate.
    """
    if (state["verification"] or {}).get("issues"):
        return "write_report"
    return "evaluate"


def next_after_review_report(state: ResearchState):
    """Route reviewer feedback to triage or an approved report to publishing.

    Args:
        state: Human report feedback after review.

    Returns:
        triage when feedback exists; otherwise publish.
    """
    if state["report_feedback"]:
        return "triage"
    return "publish"


def next_after_triage(state: ResearchState):
    """Route follow-ups needing evidence to the supervisor before rewriting.

    Args:
        state: Research request classified by the scope agent.

    Returns:
        supervisor when more research is requested; otherwise write_report.
    """
    if state["research_request"]:
        return "supervisor"
    return "write_report"
