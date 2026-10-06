"""The LangGraph state shared by every node (see ``graph/build.py`` for the flow)."""

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class GraphState(TypedDict, total=False):
    """Everything one research run carries between nodes. No channel has a reducer, so each key
    keeps the last value written: nodes return only the keys they change, and LangGraph's
    checkpointer persists the merged state per ``thread_id`` between chat turns."""

    # Identity and progress shown in the UI
    run_id: str
    checkpoint_schema_version: int
    thread_id: str
    stage: str  # a contracts.Stage value
    status: str  # a contracts.RunStatus value
    current_action: str
    activity: list[dict[str, Any]]
    last_error: str
    # Scope conversation: the latest user message, the chat history, and the scope decision
    request: str
    messages: list[BaseMessage]
    scope_assessment: dict
    scoped_request: dict  # the ResearchBrief
    # Planning and research: the plan shown for approval (hash-bound) and its approval
    plan: dict | None
    plan_version: int
    plan_approval: dict | None
    todos: list[dict]
    todo_ids: list[str]
    artifacts: list[dict]
    evidence: list[dict]
    index_refs: list[str]
    progress_assessment: dict | None
    # Build-Verify-Fix: the current draft, its verification (PASS/FAIL/ERROR), and the loop's memory
    draft: str
    previous_draft: str
    verification: dict | None
    verification_outcome: str
    verification_history: list[dict]
    coverage_cache: dict | None  # coverage checks of the current draft kept across an ERROR retry
    # Diagnosed repair plan of the open improvement cycle; its "route" is a contracts.RepairRoute
    # value (see repair.py)
    repair: dict | None
    repair_researched: list[str]  # checklist items that already received targeted research
    writer_evidence: list[dict]  # per-question ranked source ids the latest draft was written from
    # RAGAS Faithfulness is the third gate; the other RAGAS scores are informational
    faithfulness_score: float | None
    faithfulness_error: str
    context_precision_score: float | None
    context_precision_error: str
    context_precision_details: list[dict]
    response_relevancy_score: float | None
    response_relevancy_error: str
    recovery_attempt: int
    recovery_budget: int
    progress_fingerprint: str
    stall_count: int
    bvf_history: list[dict[str, Any]]
    # Human review (see ApprovalNodes): the latest reply, how it was classified, who gave it, and
    # every "revise" request since the last scope change (kept in force through re-verification)
    human_feedback: str
    feedback_route: str  # a contracts.FeedbackDecision route ("revise", "extend", "scope")
    reviewer: str
    reviewer_requests: list[str]
    # An "extend" reply: its topic, then what locate_extension_evidence found for it
    extension: dict | None
    recovery_decision: dict | None
    recovery_route: str  # a contracts.RecoveryAction value
    research_gaps: list[dict]
    # The research supervisor's ProposeReplan: {"reason", "todo_ids"}, cleared by replan
    research_replan: dict | None
    research_gap_attempt: int
    research_gap_budget: int
    verification_issue: str  # a contracts.VerificationIssue value
    verifier_attempt: int
    verifier_error: str
    draft_version: int
    research_status: str
    research_limitations: list[str]
    result_incomplete: bool
    # Publication
    approval: dict | None
    publication: dict | None


__all__ = ["GraphState"]
