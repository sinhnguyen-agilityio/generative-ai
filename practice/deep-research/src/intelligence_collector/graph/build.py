"""Graph wiring: which nodes exist and how they connect.

Node logic lives in ``nodes.py``, conditional routers in ``routing.py``, and
tracing/checkpoint/lifecycle concerns in ``instrumentation.py``.
"""

from langgraph.graph import END, START, StateGraph

from intelligence_collector.agents.planner import PlannerAgent
from intelligence_collector.graph.instrumentation import node_wrapper
from intelligence_collector.graph.nodes import (
    Node,
    PlannerNodes,
    ScopeNodes,
    approval,
    stage_only,
)
from intelligence_collector.graph.routing import (
    FIX,
    NEEDS_REVIEW,
    PASSED,
    route_after_clarify,
    route_after_fix,
    route_after_verify,
)
from intelligence_collector.graph.state import GraphState

__all__ = ["build_graph", "node_wrapper"]


def _chain(graph: StateGraph, *names: str) -> None:
    """Connect nodes with unconditional edges in the given order."""
    for source, target in zip(names, names[1:]):
        graph.add_edge(source, target)


def build_graph(
    *,
    checkpointer=None,
    checkpoint_store=None,
    with_approval: bool = False,
    scope_agent=None,
    planner_agent=None,
    research_nodes=None,
    vfs=None,
    tracer=None,
    lifecycle=None,
):
    """Build the research workflow graph.

    The topology depends on which collaborators are injected::

        Full (scope_agent + research_nodes):
            START -> clarify_with_user --(needs clarification)--> END
                                       \\-> write_research_brief -> planning
                  -> researching -> research_complete -> synthesizing -> verifying
            verifying --(all checks pass)--> approval (with_approval) | END
                      --(failed, budget left)--> fixing --(DRAFT_DEFECT)--> synthesizing
                                                        \\-(EVIDENCE_GAP)-> researching_gaps
                                                                            -> synthesizing
                      --(budget exhausted / no progress)--> needs_review -> END

        Scope only (scope_agent):
            START -> clarify_with_user -> write_research_brief -> planning -> END

        Stage-only stubs (no scope_agent, no research_nodes):
            START -> scope -> planning -> researching -> synthesizing -> verifying
                  -> [approval] -> END

    ``planning`` runs the Planner Agent: it decomposes the research brief into a list of
    research questions and persists them via ``write_todos``. ``planner_agent`` defaults to a
    ``PlannerAgent`` sharing the scope agent's model router.
    """
    graph = StateGraph(GraphState)

    def add_node(name: str, fn) -> None:
        graph.add_node(name, node_wrapper(
            name, fn, tracer, checkpoint_store, lifecycle))

    # ===== SCOPE AND PLANNING =====

    if scope_agent is not None:
        scope = ScopeNodes(scope_agent, vfs)
        planner = PlannerNodes(
            planner_agent or PlannerAgent(scope_agent.router), vfs)
        add_node(Node.CLARIFY, scope.clarify_with_user)
        add_node(Node.WRITE_BRIEF, scope.write_research_brief)
        add_node(Node.PLAN, planner.planning)

        graph.add_edge(START, Node.CLARIFY)
        graph.add_conditional_edges(Node.CLARIFY, route_after_clarify, [
                                    Node.WRITE_BRIEF, END])
        graph.add_edge(Node.WRITE_BRIEF, Node.PLAN)
    else:
        for name in (Node.STUB_SCOPE, Node.PLAN):
            add_node(name, stage_only(name))
        _chain(graph, START, Node.STUB_SCOPE, Node.PLAN)

    # ===== RESEARCH PIPELINE =====
    if research_nodes is None and scope_agent is not None:
        # Scope-only mode intentionally stops once a durable plan exists.
        graph.add_edge(Node.PLAN, END)
        return graph.compile(checkpointer=checkpointer)

    if research_nodes is None:
        for name in (Node.RESEARCH, Node.SYNTHESIZE, Node.VERIFY):
            add_node(name, stage_only(name))
        _chain(graph, Node.PLAN, Node.RESEARCH, Node.SYNTHESIZE, Node.VERIFY)
        if with_approval:
            # Stub verification produces no result, so it always proceeds to approval.
            add_node(Node.APPROVE, approval)
            _chain(graph, Node.VERIFY, Node.APPROVE, END)
        else:
            graph.add_edge(Node.VERIFY, END)
        return graph.compile(checkpointer=checkpointer)

    add_node(Node.RESEARCH, research_nodes.researching)
    add_node(Node.ASSESS_RESEARCH, research_nodes.research_complete)
    _chain(graph, Node.PLAN, Node.RESEARCH,
           Node.ASSESS_RESEARCH, Node.SYNTHESIZE)

    # ===== BUILD-VERIFY-FIX LOOP =====
    bvf = research_nodes.bvf
    add_node(Node.SYNTHESIZE, bvf.synthesizing)
    add_node(Node.VERIFY, bvf.verifying)
    add_node(Node.FIX, bvf.fixing)
    add_node(Node.RESEARCH_GAPS, bvf.researching_gaps)
    add_node(Node.NEEDS_REVIEW, bvf.needs_review)

    graph.add_edge(Node.SYNTHESIZE, Node.VERIFY)
    passed_target = Node.APPROVE if with_approval else END
    graph.add_conditional_edges(
        Node.VERIFY,
        route_after_verify,
        {PASSED: passed_target, FIX: Node.FIX, NEEDS_REVIEW: Node.NEEDS_REVIEW},
    )
    graph.add_conditional_edges(
        Node.FIX,
        route_after_fix,
        [Node.SYNTHESIZE, Node.RESEARCH_GAPS, Node.NEEDS_REVIEW],
    )
    graph.add_edge(Node.RESEARCH_GAPS, Node.SYNTHESIZE)
    graph.add_edge(Node.NEEDS_REVIEW, END)

    # ===== APPROVAL =====
    if with_approval:
        add_node(Node.APPROVE, approval)
        graph.add_edge(Node.APPROVE, END)

    return graph.compile(checkpointer=checkpointer)
