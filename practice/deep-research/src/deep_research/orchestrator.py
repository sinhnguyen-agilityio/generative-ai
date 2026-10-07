"""Orchestrator: creates the agents and wires them into the research graph.

    guardrail → clarify → write_brief → planner → review_plan ⏸ → supervisor ⇄ supervisor_tools
              → write_report → evaluate → review_report ⏸ → publish

⏸ = human review (LangGraph interrupt). Revising the plan loops back to the planner; revising
the report loops back to the writer. The evaluator only records scores and never changes the route.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from deep_research.agents.evaluator import EvaluatorAgent
from deep_research.agents.guardrail import GuardrailAgent
from deep_research.agents.human import HumanAgent
from deep_research.agents.planner import PlannerAgent
from deep_research.agents.publisher import PublisherAgent
from deep_research.agents.researcher import ResearcherAgent
from deep_research.agents.scope import ScopeAgent
from deep_research.agents.supervisor import SupervisorAgent
from deep_research.agents.writer import WriterAgent
from deep_research.config import Settings
from deep_research.state import ResearchState


def build_graph(settings: Settings, reasoning, standard, search, store, checkpointer=None):
    """``reasoning`` and ``standard`` are model tiers (see models.Tier)."""
    guardrail = GuardrailAgent(standard)
    scope = ScopeAgent(reasoning)
    planner = PlannerAgent(reasoning)
    human = HumanAgent()
    researcher = ResearcherAgent(standard, search, store, settings)
    supervisor = SupervisorAgent(reasoning, researcher, settings)
    writer = WriterAgent(standard)
    evaluator = EvaluatorAgent(reasoning, settings)
    publisher = PublisherAgent(settings.output_dir)

    graph = StateGraph(ResearchState)
    graph.add_node("guardrail", guardrail.check)
    graph.add_node("clarify", scope.clarify)
    graph.add_node("write_brief", scope.write_brief)
    graph.add_node("planner", planner.plan)
    graph.add_node("review_plan", human.review_plan)
    graph.add_node("supervisor", supervisor.supervisor)
    graph.add_node("supervisor_tools", supervisor.supervisor_tools)
    graph.add_node("write_report", writer.write)
    graph.add_node("evaluate", evaluator.evaluate)
    graph.add_node("review_report", human.review_report)
    graph.add_node("publish", publisher.publish)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", lambda s: END if s["blocked"] else "clarify", ["clarify", END])
    graph.add_conditional_edges("clarify", lambda s: END if s["needs_clarification"] else "write_brief", ["write_brief", END])
    graph.add_edge("write_brief", "planner")
    graph.add_edge("planner", "review_plan")
    graph.add_conditional_edges("review_plan", lambda s: "planner" if s["plan_feedback"] else "supervisor", ["planner", "supervisor"])
    graph.add_conditional_edges("supervisor", supervisor.next_step, ["supervisor_tools", "write_report"])
    graph.add_edge("supervisor_tools", "supervisor")
    graph.add_edge("write_report", "evaluate")
    graph.add_edge("evaluate", "review_report")
    graph.add_conditional_edges(
        "review_report", lambda s: "write_report" if s["report_feedback"] else "publish", ["write_report", "publish"]
    )
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=checkpointer, name="deep_research")
