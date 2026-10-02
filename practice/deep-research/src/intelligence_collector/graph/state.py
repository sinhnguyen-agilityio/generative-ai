"""The LangGraph state shared by every node (see ``graph/build.py`` for the flow)."""

from typing import Any, TypedDict

from langchain_core.messages import BaseMessage


class GraphState(TypedDict, total=False):
    """Everything one research run carries between nodes. Nodes return the full updated state;
    LangGraph's checkpointer persists it per ``thread_id`` between chat turns."""

    # Identity and progress shown in the UI
    run_id: str
    thread_id: str
    stage: str  # a contracts.Stage value
    status: str
    current_action: str
    activity: list[dict[str, Any]]
    last_error: str
    # Scope conversation: the latest user message, the chat history, and the scope decision
    request: str
    messages: list[BaseMessage]
    scope_assessment: dict
    scoped_request: dict  # the ResearchBrief
    # Planning and research
    todos: list[dict]
    todo_ids: list[str]
    artifacts: list[dict]
    evidence: list[dict]
    index_refs: list[str]
    progress_assessment: dict | None
    # Build-Verify-Fix: the current draft, its verification, and the loop's memory
    draft: str
    previous_draft: str
    verification: dict | None
    verification_history: list[dict]
    # RAGAS quality scores of the final report (informational; see BVFNodes.evaluating_quality)
    writer_evidence: list[dict]  # per-question ranked source ids the latest draft was written from
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
    bvf_lessons: list[str]
    bvf_history: list[dict[str, Any]]
    fix_strategy: str
    # Human review (see ApprovalNodes): the latest reply, how it was classified, who gave it, and
    # every "revise" request since the last scope change (kept in force through re-verification)
    human_feedback: str
    feedback_route: str
    reviewer: str
    reviewer_requests: list[str]
    recovery_decision: dict | None
    # Publication
    approval: dict | None
    publication: dict | None


__all__ = ["GraphState"]
