"""Test entry points into the production graph.

``build_graph`` only wires the production topology, which always starts with scoping and plan
approval. Tests that inject a brief and todos directly start the same graph at research:
``research_graph`` writes the initial state as an approved plan, so the graph continues exactly
as it would after a reviewer approved it. ``approval_graph`` is a bare publication interrupt for
checkpoint and recovery tests.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from intelligence_collector.graph.build import build_graph
from intelligence_collector.graph.instrumentation import node_wrapper
from intelligence_collector.graph.nodes import Node
from intelligence_collector.graph.state import GraphState

APPROVED_PLAN = {"outcome": "approve", "approved": True}


class NoScopeAgent:
    """Scope agent for graphs entered at research: scoping never runs, so it has no model."""

    router = None


class ResearchStartGraph:
    """A compiled production graph whose dict input starts at research instead of scoping.

    The input is recorded as the update of an approved ``await_plan_approval``; a ``Command`` or
    ``None`` resumes as usual. Without a Planner or plan record the run behaves like a fixed
    plan: no replacement planning and no rescoping (`research_graph`)."""

    def __init__(self, graph):
        self.graph = graph

    def invoke(self, value, config=None, **kwargs):
        if isinstance(value, dict):
            config = config or {
                "configurable": {"thread_id": value.get("thread_id") or value["run_id"]}
            }
            # Seeding is setup, not workflow: keep it out of the caller's traces.
            self.graph.update_state(
                {"configurable": config["configurable"]},
                {**value, "plan_approval": APPROVED_PLAN},
                as_node=Node.AWAIT_PLAN,
            )
            value = None
        return self.graph.invoke(value, config, **kwargs)

    def __getattr__(self, name):
        return getattr(self.graph, name)


def research_graph(research_nodes, *, checkpointer=None, **kwargs) -> ResearchStartGraph:
    """The production graph around `research_nodes`, entered at research with injected todos."""
    graph = build_graph(
        scope_agent=NoScopeAgent(),
        research_nodes=research_nodes,
        checkpointer=checkpointer or MemorySaver(),
        **kwargs,
    )
    # Injected todos are a fixed plan: there is no Planner to replace it.
    research_nodes.supervisor.replan_enabled = False
    research_nodes.bvf.planner = None
    return ResearchStartGraph(graph)


def approval(state: GraphState) -> GraphState:
    """Pause until a human resumes with a publication decision (no report is published)."""
    from langgraph.types import interrupt

    decision = interrupt({"kind": "publication_approval", "run_id": state.get("run_id")})
    stage = "Published" if decision.get("approved") else "Awaiting Approval"
    return {**state, "approval": decision, "stage": stage}


def approval_graph(*, checkpointer, run_registry=None):
    """START -> approval (publication interrupt) -> END."""
    builder = StateGraph(GraphState)
    builder.add_node("approval", node_wrapper("approval", approval, run_registry=run_registry))
    builder.add_edge(START, "approval")
    builder.add_edge("approval", END)
    return builder.compile(checkpointer=checkpointer)
