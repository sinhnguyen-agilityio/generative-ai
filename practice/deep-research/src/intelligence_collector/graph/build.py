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
    await_plan_approval,
    await_recovery,
    stage_only,
)
from intelligence_collector.graph.routing import (
    APPROVED,
    FIX,
    NEEDS_REVIEW,
    PASSED,
    PLAN_APPROVED,
    PLAN_CHANGE,
    PLAN_DECLINED,
    REJECTED,
    REPLACEMENT_DECLINED,
    TO_EXTEND,
    TO_PUBLISH,
    TO_REVISE,
    TO_SCOPE,
    route_after_approval,
    route_after_clarify,
    route_after_extension,
    route_after_feedback,
    route_after_fix,
    route_after_plan_approval,
    route_after_verify,
    route_after_recovery,
    route_after_research,
)
from intelligence_collector.graph.state import GraphState
from intelligence_collector.graph.operation_journal import OperationJournal

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
    extension_judge=None,
):
    """Build the research workflow graph.

    The topology depends on which collaborators are injected::

        Full (scope_agent + research_nodes):
            START -> clarify_with_user --(needs clarification)--> await_clarification (interrupt)
                                       <--(validated human reply)--/
                                       \\-> write_research_brief -> planning
                  -> await_plan_approval (interrupt; nothing is retrieved before approval)
                  -> researching -> research_complete -> synthesizing -> verifying
            await_plan_approval --(change)--> clarify_with_user  --(decline)--> END
            verifying --(PASS: coverage, citations, Faithfulness > 0.90)--> evaluating_quality
                          -> approval (with_approval) | END
                      --(FAIL, cycle budget left)--> fixing (diagnosis)
                          -> synthesizing (vfs / chroma / presentation repair)
                          | prepare_repair_research -> researching (targeted Tavily research)
                          | replan -> await_plan_approval (replacement strategy)
                      --(ERROR)--> retry_verifier (no rewrite) | needs_review
                      --(budget exhausted / no progress)--> needs_review
                          -> await_recovery (interrupt) -> apply_recovery_decision
                          -> authorized repair, research, rescope, export, or cancellation
            approval (human-in-the-loop review) --(explicit approve)--> publishing -> END
                     --(chat reply / reject)--> classify_feedback (LLM reads the reply)
            classify_feedback --(approve)--> publishing -> END
                              --(revise)--> synthesizing (rewrite with the request, re-verified)
                              --(extend)--> locate_extension_evidence
                                  --(stored evidence enough)--> synthesizing
                                  --(missing points)--> researching (only the added task)
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

    bvf = getattr(research_nodes, "bvf", None)
    writer = getattr(bvf, "writer", None)
    journal_context = repr((getattr(bvf, "report_guidelines", None),
                            getattr(bvf, "evidence_top_k", None),
                            getattr(getattr(writer, "router", None), "model_name", None)))
    journal = OperationJournal(vfs, journal_context) if vfs is not None else None

    def add_node(name: str, fn) -> None:
        """Register a node wrapped with tracing and lifecycle notifications."""
        graph.add_node(name, node_wrapper(
            name, fn, tracer, lifecycle, run_registry,
            journal if name == Node.SYNTHESIZE else None))

    # ===== SCOPE AND PLANNING =====

    planner_agent = planner_agent or (PlannerAgent(scope_agent.router) if scope_agent else None)
    if scope_agent is not None:
        scope = ScopeNodes(scope_agent, vfs)
        planner = PlannerNodes(planner_agent, vfs)
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
    if scope_agent is not None:
        # A real plan exists: retrieval waits for its explicit approval.
        add_node(Node.AWAIT_PLAN, await_plan_approval)
        graph.add_edge(Node.PLAN, Node.AWAIT_PLAN)
        graph.add_conditional_edges(Node.AWAIT_PLAN, route_after_plan_approval, {
            PLAN_APPROVED: Node.RESEARCH, PLAN_CHANGE: Node.CLARIFY, PLAN_DECLINED: END,
            REPLACEMENT_DECLINED: Node.NEEDS_REVIEW})
    else:
        # Stage-only planning stub: todos are injected by the caller, no plan record exists.
        graph.add_edge(Node.PLAN, Node.RESEARCH)
    graph.add_edge(Node.RESEARCH, Node.ASSESS_RESEARCH)
    graph.add_conditional_edges(Node.ASSESS_RESEARCH, route_after_research, {
        Node.SYNTHESIZE: Node.SYNTHESIZE,
        Node.PREPARE_GAP_RESEARCH: Node.PREPARE_GAP_RESEARCH,
        Node.NEEDS_REVIEW: Node.NEEDS_REVIEW,
        Node.RETRY_RESEARCH_ASSESSMENT: Node.RETRY_RESEARCH_ASSESSMENT})
    add_node(Node.RETRY_RESEARCH_ASSESSMENT, lambda state: state)
    graph.add_edge(Node.RETRY_RESEARCH_ASSESSMENT, Node.ASSESS_RESEARCH)
    add_node(Node.PREPARE_GAP_RESEARCH, research_nodes.prepare_gap_research)
    graph.add_edge(Node.PREPARE_GAP_RESEARCH, Node.RESEARCH)

    # ===== BUILD-VERIFY-FIX LOOP =====
    bvf = research_nodes.bvf
    bvf.planner = planner_agent if scope_agent is not None else None
    bvf.extension_judge = extension_judge
    add_node(Node.SYNTHESIZE, bvf.synthesizing)
    add_node(Node.VERIFY, bvf.verifying)
    add_node(Node.EVALUATE_QUALITY, bvf.evaluating_quality)
    add_node(Node.FIX, bvf.fixing)
    add_node(Node.PREPARE_REPAIR_RESEARCH, bvf.prepare_repair_research)
    graph.add_edge(Node.PREPARE_REPAIR_RESEARCH, Node.RESEARCH)
    fix_targets = [Node.SYNTHESIZE, Node.PREPARE_REPAIR_RESEARCH]
    if scope_agent is not None:
        add_node(Node.REPLAN, bvf.replan)
        graph.add_edge(Node.REPLAN, Node.AWAIT_PLAN)
        fix_targets.append(Node.REPLAN)
    add_node(Node.NEEDS_REVIEW, bvf.needs_review)
    add_node(Node.AWAIT_RECOVERY, lambda state: await_recovery(
        state, can_rescope=scope_agent is not None))
    recovery = RecoveryNodes(vfs or research_nodes.vfs, can_rescope=scope_agent is not None)
    add_node(Node.APPLY_RECOVERY, recovery.apply)
    add_node(Node.RETRY_VERIFIER, lambda state: state)
    graph.add_edge(Node.RETRY_VERIFIER, Node.VERIFY)

    graph.add_edge(Node.SYNTHESIZE, Node.VERIFY)
    passed_target = Node.APPROVE if with_approval else END
    # Optional quality observations run only on this PASSED edge, on the exact report that
    # left the loop; a passing first draft goes straight there without an improvement cycle.
    graph.add_conditional_edges(
        Node.VERIFY,
        route_after_verify,
        {PASSED: Node.EVALUATE_QUALITY, FIX: Node.FIX, NEEDS_REVIEW: Node.NEEDS_REVIEW,
         Node.RETRY_VERIFIER: Node.RETRY_VERIFIER},
    )
    graph.add_edge(Node.EVALUATE_QUALITY, passed_target)
    graph.add_conditional_edges(Node.FIX, route_after_fix, fix_targets)
    graph.add_edge(Node.NEEDS_REVIEW, Node.AWAIT_RECOVERY)
    graph.add_edge(Node.AWAIT_RECOVERY, Node.APPLY_RECOVERY)
    graph.add_conditional_edges(Node.APPLY_RECOVERY, route_after_recovery, {
        "retry_report": Node.SYNTHESIZE,
        "revise": Node.PREPARE_REVISION,
        "research": Node.RESEARCH,
        "rescope": Node.CLARIFY if scope_agent is not None else END,
        "export_incomplete": END,
        "cancel": END,
    })

    def prepare_revision(state):
        from intelligence_collector.agents.reviewer_checks import effective_requests

        return {**state, "reviewer_requests": effective_requests(
            list(state.get("reviewer_requests", []))), "verification": None,
            "approval": None, "publication": None, "verifier_attempt": 0,
            "faithfulness_score": None, "context_precision_score": None,
            "response_relevancy_score": None}

    add_node(Node.PREPARE_REVISION, prepare_revision)
    graph.add_edge(Node.PREPARE_REVISION, Node.SYNTHESIZE)

    # ===== APPROVAL =====
    if with_approval:
        approval_nodes = ApprovalNodes(vfs, output_dir, feedback_router)
        add_node(Node.APPROVE, approval_nodes.approval)
        add_node(Node.CLASSIFY_FEEDBACK, approval_nodes.classify_feedback)
        add_node(Node.LOCATE_EXTENSION, bvf.locate_extension_evidence)
        graph.add_conditional_edges(Node.LOCATE_EXTENSION, route_after_extension,
                                    [Node.SYNTHESIZE, Node.RESEARCH])
        add_node(Node.PUBLISH, approval_nodes.publishing)
        graph.add_conditional_edges(
            Node.APPROVE,
            route_after_approval,
            {APPROVED: Node.PUBLISH, REJECTED: Node.CLASSIFY_FEEDBACK},
        )
        graph.add_conditional_edges(Node.PUBLISH,
                                    lambda state: END if (state.get("publication") or {}).get("published")
                                    else Node.NEEDS_REVIEW, [END, Node.NEEDS_REVIEW])
        graph.add_conditional_edges(
            Node.CLASSIFY_FEEDBACK,
            route_after_feedback,
            {
                TO_PUBLISH: Node.PUBLISH,
                TO_REVISE: Node.PREPARE_REVISION,
                TO_EXTEND: Node.LOCATE_EXTENSION,
                # A scope change re-enters scoping (the stub scope node when no scope agent).
                TO_SCOPE: Node.CLARIFY if scope_agent is not None else Node.STUB_SCOPE,
            },
        )

    return graph.compile(checkpointer=checkpointer)
