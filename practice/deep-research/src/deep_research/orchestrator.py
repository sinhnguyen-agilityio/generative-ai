"""Orchestrator: creates the agents and wires them into the research graph.

    guardrail → clarify → write_brief → planner → review_plan ⏸ → supervisor (subgraph)
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
from deep_research.settings import Settings
from deep_research.states.research_state import ResearchState
from deep_research.utils.routing import (
    next_after_clarify,
    next_after_guardrail,
    next_after_review_plan,
    next_after_review_report,
    next_after_triage,
    next_after_verify,
)


def build_graph(settings: Settings, reasoning, standard, search, store, checkpointer=None, scorer=None):
    """Create the agents and compile the research workflow with human review checkpoints.

    Args:
        settings: Runtime settings, research limits, and storage paths.
        reasoning: Model tier for planning, supervision, scoping, and verification.
        standard: Model tier for guardrails, research, and report writing.
        search: Asynchronous web search backend.
        store: Page indexing and passage retrieval backend.
        checkpointer: Optional LangGraph checkpointer for thread persistence.
        scorer: Optional callable replacing the default RAGAS evaluator.

    Returns:
        Compiled graph from request validation through report publication.
    """
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
    graph.add_node("supervisor", supervisor.graph)
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
    graph.add_edge("supervisor", "write_report")
    graph.add_edge("write_report", "verify")
    graph.add_conditional_edges(
        "verify", next_after_verify, ["write_report", "evaluate"]
    )
    graph.add_edge("evaluate", "review_report")
    graph.add_conditional_edges("review_report", next_after_review_report, ["triage", "publish"])
    graph.add_conditional_edges("triage", next_after_triage, ["supervisor", "write_report"])
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=checkpointer, name="deep_research")
