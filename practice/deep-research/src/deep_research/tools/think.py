"""Strategic reflection tool for the supervisor and researchers."""

from langchain_core.tools import tool


@tool("Think", parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Record your assessment of research progress and choose the next action.

    Use before delegation when there is no approved initial plan, and after each
    round of research or search results before continuing or finishing. Call alone
    after results arrive, never alongside the calls whose results you are assessing.
    Cover established findings and supporting sources, missing essential information,
    evidence quality and sufficiency, and the exact next topic/query or decision to stop.
    This tool records the assessment you provide; it fetches no evidence and calls no model.

    Args:
        reflection: Your assessment of findings, gaps, evidence quality, and next action.

    Returns:
        Confirmation containing your reflection for the next model decision.
    """
    return f"Reflection recorded: {reflection}"
