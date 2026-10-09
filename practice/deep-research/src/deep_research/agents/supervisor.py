"""Supervisor agent: researches the approved plan, reviews the topic reports and decides whether
to delegate follow-ups or hand over to the writer. It also takes the reviewer's requests for more
research on a written report.

    START → supervisor ⇄ supervisor_tools → END
"""

from __future__ import annotations

import asyncio
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from deep_research.agents.researcher import ResearcherAgent, render_note
from deep_research.agents.utils import bullets, today
from deep_research.config import Settings
from deep_research.prompts import SUPERVISOR_MORE, SUPERVISOR_PROMPT, SUPERVISOR_START
from deep_research.state import ConductResearch, ResearchComplete, SupervisorInput, SupervisorOutput, SupervisorState, Think


class SupervisorAgent:
    def __init__(self, llm, researcher: ResearcherAgent, settings: Settings):
        self.llm = llm.tools([ConductResearch, ResearchComplete, Think])
        self.researcher = researcher
        self.max_topics_per_round = settings.max_concurrent_researchers
        self.max_iterations = settings.max_supervisor_iterations
        self.graph = self._build()

    async def supervisor(self, state: SupervisorState):
        """Let the model delegate topics, review findings, or complete research.

        An initial plan allows direct delegation; without one the model plans with Think.
        Follow-ups reopen the conversation after answering the previous closing calls.

        Args:
            state: Brief, optional plan, research history, and any follow-up request.
        Returns:
            Updated messages, model-turn count, and a cleared follow-up request.
        """
        messages = state.get("supervisor_messages") or []
        added = []
        if not messages:
            added.append(HumanMessage(SUPERVISOR_START.format(
                brief=state["brief"], plan=bullets(state.get("plan") or []) or "None."
            )))
        elif state.get("research_request"):
            added = [
                ToolMessage("Closed: the report was written.", tool_call_id=call["id"], name=call["name"])
                for call in messages[-1].tool_calls
            ]
            added.append(HumanMessage(SUPERVISOR_MORE.format(request=state["research_request"])))
        system = SystemMessage(SUPERVISOR_PROMPT.format(date=today(), max_parallel=self.max_topics_per_round, max_iterations=self.max_iterations))
        response = await self.llm.ainvoke([system, *messages, *added])
        return {
            "supervisor_messages": [*added, response],
            "supervisor_iterations": state.get("supervisor_iterations", 0) + 1,
            "research_request": "",
        }

    def next_step(self, state: SupervisorState) -> Literal["supervisor_tools", "__end__"]:
        """Route the latest model decision to tool execution or subgraph completion.

        Args:
            state: Latest supervisor response and number of model turns.
        Returns:
            END on completion, no tool calls, or the iteration limit; otherwise tools.
        """
        calls = state["supervisor_messages"][-1].tool_calls
        exceeded_iterations = state.get("supervisor_iterations", 0) >= self.max_iterations
        no_tool_calls = not calls
        research_complete = any(call["name"] == "ResearchComplete" for call in calls)
        return END if exceeded_iterations or no_tool_calls or research_complete else "supervisor_tools"

    async def supervisor_tools(self, state: SupervisorState):
        """Execute delegated research concurrently and acknowledge reflection calls.

        Args:
            state: Latest supervisor message containing tool calls.
        Returns:
            A ToolMessage per call and new topic notes after the entire round finishes.
        """
        calls = state["supervisor_messages"][-1].tool_calls
        research = [call for call in calls if call["name"] == "ConductResearch"]
        results = await asyncio.gather(*(self.researcher.research(call["args"]["research_topic"]) for call in research))
        notes = {call["id"]: note for call, note in zip(research, results)}
        replies = []
        for call in calls:
            if call["name"] == "ConductResearch":
                content = render_note(notes[call["id"]])
            elif call["name"] == "Think":
                content = "Reflection recorded."
            else:
                content = f"Unknown tool: {call['name']}"
            replies.append(ToolMessage(content, tool_call_id=call["id"], name=call["name"]))
        return {"supervisor_messages": replies, "notes": list(notes.values())}

    def _build(self):
        """Return the supervisor/tools subgraph with parent checkpoint inheritance."""
        builder = StateGraph(SupervisorState, input_schema=SupervisorInput, output_schema=SupervisorOutput)
        builder.add_node("supervisor", self.supervisor)
        builder.add_node("supervisor_tools", self.supervisor_tools)
        builder.add_edge(START, "supervisor")
        builder.add_conditional_edges("supervisor", self.next_step, ["supervisor_tools", END])
        builder.add_edge("supervisor_tools", "supervisor")
        # Inherit the parent's checkpointer so interrupted runs can resume inside this graph.
        return builder.compile(name="supervisor")
