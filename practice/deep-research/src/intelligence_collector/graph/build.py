"""Graph wiring: which nodes exist and how they connect.

Node logic lives in ``nodes.py``, conditional routers in ``routing.py``, and
tracing/checkpoint/lifecycle concerns in ``instrumentation.py``.
"""

from langgraph.graph import END, START, StateGraph

from intelligence_collector.agents.planner import PlannerAgent
from intelligence_collector.graph.instrumentation import node_wrapper
from intelligence_collector.graph.nodes import (
    ApprovalNodes,
    Node,
    PlannerNodes,
    ScopeNodes,
    RecoveryNodes,
    approval,
    await_clarification,
    await_recovery,
    stage_only,
)
from intelligence_collector.graph.routing import (
    APPROVED,
    FIX,
    NEEDS_REVIEW,
    PASSED,
    REJECTED,
    TO_PUBLISH,
    TO_REVISE,
    TO_SCOPE,
    route_after_approval,
    route_after_clarify,
    route_after_feedback,
    route_after_verify,
    route_after_recovery,
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
    with_approval: bool = False,
    scope_agent=None,
    planner_agent=None,
    research_nodes=None,
    vfs=None,
    tracer=None,
    lifecycle=None,
    run_registry=None,
    output_dir: str = "exports",
    feedback_router=None,
):
    """Build the research workflow graph.

    The topology depends on which collaborators are injected::

        Full (scope_agent + research_nodes):
            START -> clarify_with_user --(needs clarification)--> await_clarification (interrupt)
                                       <--(validated human reply)--/
                                       \\-> write_research_brief -> planning
                  -> researching -> research_complete -> synthesizing -> verifying
            verifying --(all checks pass)--> evaluating_quality -> approval (with_approval) | END
                      --(failed, budget left)--> fixing (reflect) -> synthesizing (rewrite)
                      --(budget exhausted / no progress)--> needs_review
                          -> await_recovery (interrupt) -> apply_recovery_decision
                          -> authorized repair, research, rescope, export, or cancellation
            approval (human-in-the-loop review) --(explicit approve)--> publishing -> END
                     --(chat reply / reject)--> classify_feedback (LLM reads the reply)
            classify_feedback --(approve)--> publishing -> END
                              --(revise)--> synthesizing (rewrite with the request, re-verified)
                              --(scope)--> clarify_with_user -> brief -> planning
                                           (fresh plan, discards prior draft/verification)

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
        """Register a node wrapped with tracing and lifecycle notifications."""
        graph.add_node(name, node_wrapper(
            name, fn, tracer, lifecycle, run_registry))

    # ===== SCOPE AND PLANNING =====

    if scope_agent is not None:
        scope = ScopeNodes(scope_agent, vfs)
        planner = PlannerNodes(
            planner_agent or PlannerAgent(scope_agent.router), vfs)
        add_node(Node.CLARIFY, scope.clarify_with_user)
        add_node(Node.AWAIT_CLARIFICATION, await_clarification)
        add_node(Node.WRITE_BRIEF, scope.write_research_brief)
        add_node(Node.PLAN, planner.planning)

        graph.add_edge(START, Node.CLARIFY)
        graph.add_conditional_edges(Node.CLARIFY, route_after_clarify, [
                                    Node.WRITE_BRIEF, Node.AWAIT_CLARIFICATION])
        graph.add_edge(Node.AWAIT_CLARIFICATION, Node.CLARIFY)
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
    add_node(Node.EVALUATE_QUALITY, bvf.evaluating_quality)
    add_node(Node.FIX, bvf.fixing)
    add_node(Node.NEEDS_REVIEW, bvf.needs_review)
    add_node(Node.AWAIT_RECOVERY, lambda state: await_recovery(
        state, can_rescope=scope_agent is not None))
    recovery = RecoveryNodes(vfs or research_nodes.vfs, can_rescope=scope_agent is not None)
    add_node(Node.APPLY_RECOVERY, recovery.apply)

    graph.add_edge(Node.SYNTHESIZE, Node.VERIFY)
    passed_target = Node.APPROVE if with_approval else END
    # RAGAS quality scoring (evaluating_quality) runs only on this PASSED edge — once, on the
    # final report that actually left the BVF loop — never on a draft that gets fixed and
    # reverified.
    graph.add_conditional_edges(
        Node.VERIFY,
        route_after_verify,
        {PASSED: Node.EVALUATE_QUALITY, FIX: Node.FIX, NEEDS_REVIEW: Node.NEEDS_REVIEW},
    )
    graph.add_edge(Node.EVALUATE_QUALITY, passed_target)
    # The fix is always a rewrite from the same evidence.
    graph.add_edge(Node.FIX, Node.SYNTHESIZE)
    graph.add_edge(Node.NEEDS_REVIEW, Node.AWAIT_RECOVERY)
    graph.add_edge(Node.AWAIT_RECOVERY, Node.APPLY_RECOVERY)
    graph.add_conditional_edges(Node.APPLY_RECOVERY, route_after_recovery, {
        "retry_report": Node.SYNTHESIZE,
        "revise": Node.SYNTHESIZE,
        "research": Node.RESEARCH,
        "rescope": Node.CLARIFY if scope_agent is not None else END,
        "export_incomplete": END,
        "cancel": END,
    })

    # ===== APPROVAL =====
    if with_approval:
        approval_nodes = ApprovalNodes(vfs, output_dir, feedback_router)
        add_node(Node.APPROVE, approval_nodes.approval)
        add_node(Node.CLASSIFY_FEEDBACK, approval_nodes.classify_feedback)
        add_node(Node.PUBLISH, approval_nodes.publishing)
        graph.add_conditional_edges(
            Node.APPROVE,
            route_after_approval,
            {APPROVED: Node.PUBLISH, REJECTED: Node.CLASSIFY_FEEDBACK},
        )
        graph.add_edge(Node.PUBLISH, END)
        graph.add_conditional_edges(
            Node.CLASSIFY_FEEDBACK,
            route_after_feedback,
            {
                TO_PUBLISH: Node.PUBLISH,
                TO_REVISE: Node.SYNTHESIZE,
                # A scope change re-enters scoping (the stub scope node when no scope agent).
                TO_SCOPE: Node.CLARIFY if scope_agent is not None else Node.STUB_SCOPE,
            },
        )

    return graph.compile(checkpointer=checkpointer)
