"""Node identifiers, workflow limits and the activity-log helper shared by all nodes."""

from __future__ import annotations

DEFAULT_RECOVERY_BUDGET = 3
# RAGAS Faithfulness must be strictly greater than this for the third gate to pass.
FAITHFULNESS_THRESHOLD = 0.90
# Identical failures on this many consecutive re-verifications, despite reflection, ends the loop.
MAX_STALLED_ATTEMPTS = 2
# Stored passages the extension judge reads for a reviewer's "extend" request.
EXTENSION_PASSAGES = 8


class Node:
    """Stable node identifiers shared by graph wiring, UI progress, tracing and tests."""

    # Scope and planning
    CLARIFY = "clarify_with_user"
    AWAIT_CLARIFICATION = "await_clarification"
    WRITE_BRIEF = "write_research_brief"
    PLAN = "planning"
    AWAIT_PLAN = "await_plan_approval"
    # Research pipeline
    RESEARCH = "researching"
    ASSESS_RESEARCH = "research_complete"
    # Build-Verify-Fix loop
    SYNTHESIZE = "synthesizing"
    PREPARE_REVISION = "prepare_revision"
    LOCATE_EXTENSION = "locate_extension_evidence"
    VERIFY = "verifying"
    EVALUATE_QUALITY = "evaluating_quality"
    FIX = "fixing"
    PREPARE_REPAIR_RESEARCH = "prepare_repair_research"
    REPLAN = "replan"
    NEEDS_REVIEW = "needs_review"
    AWAIT_RECOVERY = "await_recovery"
    APPLY_RECOVERY = "apply_recovery_decision"
    PREPARE_GAP_RESEARCH = "prepare_gap_research"
    RETRY_VERIFIER = "retry_verifier"
    RETRY_RESEARCH_ASSESSMENT = "retry_research_assessment"
    # Human review and publication
    APPROVE = "approval"
    CLASSIFY_FEEDBACK = "classify_feedback"
    PUBLISH = "publishing"
    # Placeholder used by the stage-only graph when no Scope Agent is injected
    STUB_SCOPE = "scope"


# Agent (process) that owns each node: shown in the UI activity log and in Langfuse span names.
NODE_AGENT = {
    Node.CLARIFY: "Scope Agent",
    Node.AWAIT_CLARIFICATION: "Scope Agent",
    Node.WRITE_BRIEF: "Scope Agent",
    Node.STUB_SCOPE: "Scope Agent",
    Node.PLAN: "Planner",
    Node.AWAIT_PLAN: "Planner",
    Node.REPLAN: "Planner",
    Node.RESEARCH: "Research Supervisor",
    Node.PREPARE_GAP_RESEARCH: "Research Supervisor",
    Node.PREPARE_REPAIR_RESEARCH: "Research Supervisor",
    Node.ASSESS_RESEARCH: "Research Evidence Judge",
    Node.RETRY_RESEARCH_ASSESSMENT: "Research Evidence Judge",
    Node.SYNTHESIZE: "Report Writer",
    Node.PREPARE_REVISION: "Report Writer",
    Node.LOCATE_EXTENSION: "Report Writer",
    Node.VERIFY: "Verifier",
    Node.EVALUATE_QUALITY: "Verifier",
    Node.RETRY_VERIFIER: "Verifier",
    Node.NEEDS_REVIEW: "Verifier",
    Node.AWAIT_RECOVERY: "Verifier",
    Node.APPLY_RECOVERY: "Verifier",
    Node.FIX: "Repair Diagnosis",
    Node.CLASSIFY_FEEDBACK: "Feedback Router",
    Node.APPROVE: "Publisher",
    Node.PUBLISH: "Publisher",
}


def node_agent(node: str) -> str:
    """The agent that owns `node`; ``Workflow`` for nodes outside the mapping."""
    return NODE_AGENT.get(node, "Workflow")


def _activity(state, label: str, status: str = "success"):
    return [
        *state.get("activity", []),
        {"id": f"{len(state.get('activity', [])) + 1}:{label}", "label": label, "status": status},
    ]
