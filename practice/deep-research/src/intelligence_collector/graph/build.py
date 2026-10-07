"""Graph wiring: which nodes exist and how they connect.

Node logic lives in ``nodes.py``, conditional routers in ``routing.py``, and
tracing/checkpoint/lifecycle concerns in ``instrumentation.py``.
"""

from langgraph.graph import END, START, StateGraph

from intelligence_collector.agents.planner import PlannerAgent
from intelligence_collector.graph.instrumentation import node_wrapper
from intelligence_collector.contracts import RecoveryAction
from intelligence_collector.graph.nodes import (
    ApprovalNodes,
    Node,
    PlannerNodes,
    ScopeNodes,
    RecoveryNodes,
    await_clarification,
    await_plan_approval,
    await_recovery,
)
from intelligence_collector.graph.routing import (
    APPROVED,
    FIX,
    NEEDS_REVIEW,
    PASSED,
    PLAN_APPROVED,
    PLAN_CHANGE,
    PLAN_DECLINED,
    PUBLISHED,
    REJECTED,
    REPLACEMENT_DECLINED,
    TO_EXTEND,
    TO_REVISE,
    TO_SCOPE,
    route_after_approval,
    route_after_clarify,
    route_after_delegation,
    route_after_extension,
    route_after_feedback,
    route_after_fix,
    route_after_gap_analysis,
    route_after_plan_approval,
    route_after_publish,
    route_after_verify,
    route_after_recovery,
    route_after_research,
)
from intelligence_collector.graph.state import GraphState
from intelligence_collector.graph.operation_journal import OperationJournal

__all__ = ["build_graph", "node_wrapper"]


def build_graph(
    *,
    checkpointer=None,
    with_approval: bool = False,
    scope_agent,
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
    """Build the production research workflow graph.

    ``scope_agent`` is required. The topology depends on whether ``research_nodes`` is
    injected::

        Full (scope_agent + research_nodes):
            START -> clarify_with_user --(needs clarification)--> await_clarification (interrupt)
                                       <--(validated human reply)--/
                                       \\-> write_research_brief -> planning
                  -> await_plan_approval (interrupt; nothing is retrieved before approval)
                  -> researching (supervisor LLM) -> research_complete -> synthesizing -> verifying
            researching --(supervisor ProposeReplan)--> replan -> await_plan_approval
            await_plan_approval --(change)--> clarify_with_user  --(decline)--> END
            verifying --(PASS: coverage, citations, Faithfulness > threshold)--> evaluating_quality
                          -> approval (with_approval) | END
                      --(FAIL, cycle budget left)--> fixing (diagnosis)
                          -> synthesizing (vfs / chroma / presentation repair)
                          | prepare_repair_research -> researching (targeted Tavily research)
                          | replan -> await_plan_approval (replacement strategy)
                      --(ERROR)--> retry_verifier (no rewrite) | needs_review
                      --(budget exhausted / no progress)--> needs_review
                          -> await_recovery (interrupt) -> apply_recovery_decision
                          -> authorized repair, research, rescope, export, or cancellation
            approval (human-in-the-loop review) --(Approve button)--> publishing -> END
                     --(chat reply / reject)--> classify_feedback (LLM reads the reply)
            classify_feedback --(revise)--> synthesizing (rewrite with the request, re-verified)
                              --(extend)--> locate_extension_evidence
                                  --(stored evidence enough)--> synthesizing
                                  --(missing points)--> researching (only the added task)
                              --(scope)--> clarify_with_user -> brief -> planning
                                           (fresh plan, discards prior draft/verification)

        Scope only (no research_nodes, e.g. Tavily not configured):
            START -> clarify_with_user -> write_research_brief -> planning -> END

    ``planning`` runs the Planner Agent: it decomposes the research brief into a list of
    research questions and persists them via ``write_todos``. ``planner_agent`` defaults to a
    ``PlannerAgent`` sharing the scope agent's model router. The Faithfulness threshold is
    ``contracts.FAITHFULNESS_THRESHOLD``.
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

    if scope_agent is None:
        raise ValueError("build_graph requires a scope_agent")
    planner_agent = planner_agent or PlannerAgent(scope_agent.router)
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

    # ===== RESEARCH PIPELINE =====
    if research_nodes is None:
        # Scope-only mode intentionally stops once a durable plan exists.
        graph.add_edge(Node.PLAN, END)
        return graph.compile(checkpointer=checkpointer)

    add_node(Node.RESEARCH, research_nodes.researching)
    add_node(Node.ASSESS_RESEARCH, research_nodes.research_complete)
    # Retrieval waits for an explicit approval of the plan.
    add_node(Node.AWAIT_PLAN, await_plan_approval)
    graph.add_edge(Node.PLAN, Node.AWAIT_PLAN)
    graph.add_conditional_edges(Node.AWAIT_PLAN, route_after_plan_approval, {
        PLAN_APPROVED: Node.RESEARCH, PLAN_CHANGE: Node.CLARIFY, PLAN_DECLINED: END,
        REPLACEMENT_DECLINED: Node.NEEDS_REVIEW})
    # The supervisor may propose a replacement plan, which goes back to plan approval.
    research_nodes.supervisor.replan_enabled = True
    graph.add_conditional_edges(Node.RESEARCH, route_after_delegation,
                                [Node.REPLAN, Node.ANALYZE_GAPS])
    # Every research round ends with gap analysis; follow-up tasks start the next round.
    add_node(Node.ANALYZE_GAPS, research_nodes.analyze_gaps)
    graph.add_conditional_edges(Node.ANALYZE_GAPS, route_after_gap_analysis,
                                [Node.RESEARCH, Node.REPLAN, Node.ASSESS_RESEARCH])
    graph.add_conditional_edges(Node.ASSESS_RESEARCH, route_after_research, {
        Node.SYNTHESIZE: Node.SYNTHESIZE,
        Node.PREPARE_GAP_RESEARCH: Node.PREPARE_GAP_RESEARCH,
        Node.NEEDS_REVIEW: Node.NEEDS_REVIEW})
    add_node(Node.PREPARE_GAP_RESEARCH, research_nodes.prepare_gap_research)
    graph.add_edge(Node.PREPARE_GAP_RESEARCH, Node.RESEARCH)

    # ===== BUILD-VERIFY-FIX LOOP =====
    bvf = research_nodes.bvf
    bvf.planner = planner_agent
    bvf.extension_judge = extension_judge
    add_node(Node.SYNTHESIZE, bvf.synthesizing)
    add_node(Node.VERIFY, bvf.verifying)
    add_node(Node.EVALUATE_QUALITY, bvf.evaluating_quality)
    add_node(Node.FIX, bvf.fixing)
    add_node(Node.PREPARE_REPAIR_RESEARCH, bvf.prepare_repair_research)
    graph.add_edge(Node.PREPARE_REPAIR_RESEARCH, Node.RESEARCH)
    add_node(Node.REPLAN, bvf.replan)
    graph.add_edge(Node.REPLAN, Node.AWAIT_PLAN)
    add_node(Node.NEEDS_REVIEW, bvf.needs_review)
    add_node(Node.AWAIT_RECOVERY, await_recovery)
    recovery = RecoveryNodes(vfs or research_nodes.vfs)
    add_node(Node.APPLY_RECOVERY, recovery.apply)
    # Nothing to change: verifying re-runs on the same draft with its saved coverage checks.
    add_node(Node.RETRY_VERIFIER, lambda state: {})
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
    graph.add_conditional_edges(Node.FIX, route_after_fix,
                                [Node.SYNTHESIZE, Node.PREPARE_REPAIR_RESEARCH, Node.REPLAN])
    graph.add_edge(Node.NEEDS_REVIEW, Node.AWAIT_RECOVERY)
    graph.add_edge(Node.AWAIT_RECOVERY, Node.APPLY_RECOVERY)
    graph.add_conditional_edges(Node.APPLY_RECOVERY, route_after_recovery, {
        RecoveryAction.RETRY_REPORT.value: Node.SYNTHESIZE,
        RecoveryAction.REVISE.value: Node.PREPARE_REVISION,
        RecoveryAction.RESEARCH.value: Node.RESEARCH,
        RecoveryAction.RESCOPE.value: Node.CLARIFY,
        RecoveryAction.EXPORT_INCOMPLETE.value: END,
        RecoveryAction.CANCEL.value: END,
    })

    add_node(Node.PREPARE_REVISION, bvf.prepare_revision)
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
        graph.add_conditional_edges(Node.PUBLISH, route_after_publish,
                                    {PUBLISHED: END, Node.NEEDS_REVIEW: Node.NEEDS_REVIEW})
        graph.add_conditional_edges(
            Node.CLASSIFY_FEEDBACK,
            route_after_feedback,
            {
                TO_REVISE: Node.PREPARE_REVISION,
                TO_EXTEND: Node.LOCATE_EXTENSION,
                # A scope change re-enters scoping.
                TO_SCOPE: Node.CLARIFY,
            },
        )

    return graph.compile(checkpointer=checkpointer)
