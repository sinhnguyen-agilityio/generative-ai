"""Tool schemas the research LLMs can call, in OpenAI function format.

* Research sub-agent: ``tavily_search`` and ``think_tool``.
* Research supervisor: ``ConductResearch``, ``ResearchComplete``, ``ProposeReplan`` and
  ``think_tool``.

The agents execute the calls themselves (``researcher.py``, ``supervisor.py``); these are only the
contracts the models see.
"""

from __future__ import annotations


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


SEARCH_TOOL_NAME = "tavily_search"
THINK_TOOL_NAME = "think_tool"
CONDUCT_RESEARCH_TOOL_NAME = "ConductResearch"
RESEARCH_COMPLETE_TOOL_NAME = "ResearchComplete"
PROPOSE_REPLAN_TOOL_NAME = "ProposeReplan"

SEARCH_TOOL = _function(
    SEARCH_TOOL_NAME,
    "Search the web through the approved Tavily MCP server. Returns titles, URLs, and a summary "
    "with verbatim key excerpts for the most relevant pages.",
    {"query": {"type": "string", "description": "A focused web search query."}},
    ["query"],
)

THINK_TOOL = _function(
    THINK_TOOL_NAME,
    """Tool for strategic reflection on research progress and decision-making.

Use this tool after each search to analyze results and plan next steps systematically.
This creates a deliberate pause in the research workflow for quality decision-making.

When to use:
- After receiving search results: What key information did I find?
- Before deciding next steps: Do I have enough to answer comprehensively?
- When assessing research gaps: What specific information am I still missing?
- Before concluding research: Can I provide a complete answer now?

Reflection should address:
1. Analysis of current findings - What concrete information have I gathered?
2. Gap assessment - What crucial information is still missing?
3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
4. Strategic decision - Should I continue searching or provide my answer?

Returns confirmation that reflection was recorded for decision-making.""",
    {
        "reflection": {
            "type": "string",
            "description": "Your detailed reflection on research progress, findings, gaps, and "
            "next steps.",
        }
    },
    ["reflection"],
)

CONDUCT_RESEARCH_TOOL = _function(
    CONDUCT_RESEARCH_TOOL_NAME,
    "Delegate one approved research task to a dedicated research sub-agent.",
    {
        "todo_id": {
            "type": "string",
            "description": "The id of the approved task to research.",
        },
        "research_topic": {
            "type": "string",
            "description": "Standalone instructions for the sub-agent, described in high detail "
            "(at least a paragraph): what to find, for which entities, versions and time period.",
        },
    },
    ["todo_id", "research_topic"],
)

RESEARCH_COMPLETE_TOOL = _function(
    RESEARCH_COMPLETE_TOOL_NAME,
    "Indicate that research is complete and the report can be written.",
    {},
    [],
)

PROPOSE_REPLAN_TOOL = _function(
    PROPOSE_REPLAN_TOOL_NAME,
    "Stop research and ask a human to approve a replacement plan, because the approved plan "
    "cannot be answered as written.",
    {
        "reason": {"type": "string", "description": "Why the approved plan cannot work."},
        "todo_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The approved tasks whose questions must change.",
        },
    },
    ["reason", "todo_ids"],
)


def think(reflection: str) -> str:
    """Execute ``think_tool``: the reflection itself is the value, kept in the message history."""
    return f"Reflection recorded: {reflection}"
