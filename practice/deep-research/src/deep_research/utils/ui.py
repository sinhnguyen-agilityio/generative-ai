"""Describe graph progress for the Streamlit interface."""

from __future__ import annotations

NODE_LABELS = {
    "clarify": "Understood the request",
    "write_brief": "Wrote the research brief",
    "planner": "Drafted the research plan",
    "supervisor_tools": "Researchers finished a round",
    "write_report": "Wrote the report",
    "verify": "Checked the report against its sources",
    "triage": "Read your follow-up",
    "evaluate": "Evaluated the report",
    "publish": "Published the report",
}


def _describe(node: str, update: dict) -> str:
    """Describe a completed node or the topics delegated by the supervisor.

    Args:
        node: Graph node emitting the update.
        update: Node state update, including supervisor tool calls when applicable.

    Returns:
        Human-readable progress message.
    """
    if node == "supervisor":
        calls = update["supervisor_messages"][-1].tool_calls
        topics = [c["args"]["research_topic"] for c in calls if c["name"] == "ConductResearch"]
        if topics:
            return "Researching:\n" + "\n".join(f"- {t}" for t in topics)
        return "Supervisor is wrapping up research"
    return NODE_LABELS.get(node, node)
