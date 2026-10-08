"""Supervisor agent: researches the approved plan, reviews the topic reports and decides whether
to delegate follow-ups or hand over to the writer. It also takes the reviewer's requests for more
research on a written report.

    supervisor ⇄ supervisor_tools
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deep_research.agents.researcher import ResearcherAgent, render_note
from deep_research.agents.utils import bullets, today
from deep_research.config import Settings
from deep_research.prompts import SUPERVISOR_MORE, SUPERVISOR_PROMPT, SUPERVISOR_START
from deep_research.state import ConductResearch, ResearchComplete, ResearchState, Think


class SupervisorAgent:
    def __init__(self, llm, researcher: ResearcherAgent, settings: Settings):
        self.llm = llm.tools([ConductResearch, ResearchComplete, Think])
        self.researcher = researcher
        self.max_parallel = settings.max_concurrent_researchers
        self.max_iterations = settings.max_supervisor_iterations

    def supervisor(self, state: ResearchState):
        """The first turn researches the approved plan as it is, one researcher per topic, without a
        model call and without counting as an iteration. Later turns ask the model to review the
        topic reports and delegate follow-ups. A reviewer's research request reopens the research:
        the calls that closed it are answered and the request is added before the model's turn."""
        if not state.get("supervisor_messages"):
            start = HumanMessage(SUPERVISOR_START.format(brief=state["brief"], plan=bullets(state["plan"])))
            calls = [
                {"name": "ConductResearch", "args": {"research_topic": topic}, "id": f"plan-{i}", "type": "tool_call"}
                for i, topic in enumerate(state["plan"], 1)
            ]
            return {"supervisor_messages": [start, AIMessage("", tool_calls=calls)], "supervisor_iterations": 0}
        reopen = []
        if state.get("research_request"):
            reopen = [
                ToolMessage("Closed: the report was written.", tool_call_id=call["id"], name=call["name"])
                for call in state["supervisor_messages"][-1].tool_calls
            ]
            reopen.append(HumanMessage(SUPERVISOR_MORE.format(request=state["research_request"])))
        system = SystemMessage(SUPERVISOR_PROMPT.format(date=today()))
        response = self.llm.invoke([system, *state["supervisor_messages"], *reopen])
        return {
            "supervisor_messages": [*reopen, response],
            "supervisor_iterations": state["supervisor_iterations"] + 1,
            "research_request": "",
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
        """Run every requested topic through a researcher and answer each tool call. The pool is a
        queue: at most ``max_parallel`` researchers run at once, and a free one takes the next topic.
        The node returns once the whole round is done, so the supervisor reviews all its reports."""
        calls = state["supervisor_messages"][-1].tool_calls
        research = [call for call in calls if call["name"] == "ConductResearch"]
        with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
            futures = {
                # Each thread gets a copy of the context so tracing nests under this node.
                call["id"]: pool.submit(copy_context().run, self.researcher.research, call["args"]["research_topic"])
                for call in research
            }
            notes = {call_id: future.result() for call_id, future in futures.items()}
        replies = [
            ToolMessage(
                render_note(notes[call["id"]]) if call["id"] in notes else "Reflection recorded.",
                tool_call_id=call["id"],
                name=call["name"],
            )
            for call in calls
        ]
        return {"supervisor_messages": replies, "notes": list(notes.values())}
