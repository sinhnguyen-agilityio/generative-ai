"""Orchestrator: creates the agents and wires them into the research graph.

    guardrail → clarify → write_brief → planner → review_plan ⏸ → supervisor ⇄ supervisor_tools
              → write_report ⇄ verify → evaluate → review_report ⏸ → publish

⏸ = waits for the user (LangGraph interrupt). Revising the plan loops back to the planner. The
verifier sends a report with problems back to the writer (at most twice). The verified report waits
to be published; a follow-up message goes through triage, to the writer for a rewrite or to the
supervisor first for more research. The evaluator only records scores and never changes the route.
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
from deep_research.agents.verifier import VerifierAgent
from deep_research.agents.writer import WriterAgent
from deep_research.config import Settings
from deep_research.state import ResearchState


def next_after_clarify(state: ResearchState):
    if state["needs_clarification"]:
        return END
    return "write_brief"


def next_after_guardrail(state: ResearchState):
    if state["blocked"]:
        return END
    return "clarify"


def next_after_review_plan(state: ResearchState):
    if state["plan_feedback"]:
        return "planner"
    return "supervisor"


def next_after_verify(state: ResearchState):
    if (state["verification"] or {}).get("issues"):
        return "write_report"
    return "evaluate"


def next_after_review_report(state: ResearchState):
    if state["report_feedback"]:
        return "triage"
    return "publish"


def next_after_triage(state: ResearchState):
    if state["research_request"]:
        return "supervisor"
    return "write_report"


def build_graph(settings: Settings, reasoning, standard, search, store, checkpointer=None, scorer=None):
    """``reasoning`` and ``standard`` are model tiers (see models.Tier); ``scorer`` replaces the
    evaluator's RAGAS scoring (see EvaluatorAgent)."""
    guardrail = GuardrailAgent(standard)
    scope = ScopeAgent(reasoning)
    planner = PlannerAgent(reasoning)
    human = HumanAgent()
    researcher = ResearcherAgent(standard, search, store, settings)
    supervisor = SupervisorAgent(reasoning, researcher, settings)
    writer = WriterAgent(standard)
    verifier = VerifierAgent(reasoning, settings)
    evaluator = EvaluatorAgent(settings, scorer)
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
    graph.add_node("verify", verifier.verify)
    graph.add_node("evaluate", evaluator.evaluate)
    graph.add_node("review_report", human.review_report)
    graph.add_node("triage", scope.triage)
    graph.add_node("publish", publisher.publish)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges("guardrail", next_after_guardrail, ["clarify", END])
    graph.add_conditional_edges("clarify", next_after_clarify, ["write_brief", END])
    graph.add_edge("write_brief", "planner")
    graph.add_edge("planner", "review_plan")
    graph.add_conditional_edges("review_plan", next_after_review_plan, ["planner", "supervisor"])
    graph.add_conditional_edges("supervisor", supervisor.next_step, ["supervisor_tools", "write_report"])
    graph.add_edge("supervisor_tools", "supervisor")
    graph.add_edge("write_report", "verify")
    graph.add_conditional_edges(
        "verify", next_after_verify, ["write_report", "evaluate"]
    )
    graph.add_edge("evaluate", "review_report")
    graph.add_conditional_edges("review_report", next_after_review_report, ["triage", "publish"])
    graph.add_conditional_edges("triage", next_after_triage, ["supervisor", "write_report"])
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=checkpointer, name="deep_research")
