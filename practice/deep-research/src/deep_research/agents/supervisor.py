"""Supervisor agent: delegates topics to researchers in parallel, reviews their findings and
decides whether to research more or hand over to the writer.

    supervisor ⇄ supervisor_tools
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from deep_research.agents.researcher import ResearcherAgent
from deep_research.agents.utils import bullets, today
from deep_research.config import Settings
from deep_research.prompts import SUPERVISOR_PROMPT, SUPERVISOR_START
from deep_research.state import ConductResearch, ResearchComplete, ResearchState, Think


class SupervisorAgent:
    def __init__(self, llm, researcher: ResearcherAgent, settings: Settings):
        self.llm = llm.tools([ConductResearch, ResearchComplete, Think])
        self.researcher = researcher
        self.max_parallel = settings.max_concurrent_researchers
        self.max_iterations = settings.max_supervisor_iterations

    def supervisor(self, state: ResearchState):
        """Ask the model what to do next; the first turn starts from the brief and approved plan."""
        new = []
        if not state.get("supervisor_messages"):
            new.append(HumanMessage(SUPERVISOR_START.format(brief=state["brief"], plan=bullets(state["plan"]))))
        system = SystemMessage(SUPERVISOR_PROMPT.format(date=today(), max_parallel=self.max_parallel))
        response = self.llm.invoke([system, *state.get("supervisor_messages", []), *new])
        return {
            "supervisor_messages": [*new, response],
            "supervisor_iterations": state.get("supervisor_iterations", 0) + 1,
        }

    def next_step(self, state: ResearchState) -> Literal["supervisor_tools", "write_report"]:
        calls = state["supervisor_messages"][-1].tool_calls
        done = (
            not calls
            or any(call["name"] == "ResearchComplete" for call in calls)
            or state["supervisor_iterations"] >= self.max_iterations
        )
        return "write_report" if done else "supervisor_tools"

    def supervisor_tools(self, state: ResearchState):
        """Run every requested topic through a researcher, in parallel, and answer each tool call."""
        calls = state["supervisor_messages"][-1].tool_calls
        research = [call for call in calls if call["name"] == "ConductResearch"]
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futures = {
                # Each thread gets a copy of the context so tracing nests under this node.
                call["id"]: pool.submit(copy_context().run, self.researcher.research, call["args"]["research_topic"])
                for call in research
            }
            findings = {call_id: future.result() for call_id, future in futures.items()}
        replies = [
            ToolMessage(findings.get(call["id"], "Reflection recorded."), tool_call_id=call["id"], name=call["name"])
            for call in calls
        ]
        return {"supervisor_messages": replies, "notes": list(findings.values())}
