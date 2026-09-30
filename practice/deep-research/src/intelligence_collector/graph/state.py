from typing import Any, TypedDict

from intelligence_collector.contracts import ResearchGraphState
from langchain_core.messages import BaseMessage


class GraphState(TypedDict, total=False):
    run_id: str
    thread_id: str
    stage: str
    status: str
    payload: dict
    approval: dict
    request: str
    messages: list[BaseMessage]
    scope_assessment: dict
    scoped_request: dict
    todos: list[dict]
    todo_ids: list[str]
    artifacts: list[dict]
    evidence: list[dict]
    index_refs: list[str]
    progress_assessment: dict | None
    draft: str
    previous_draft: str
    verification: dict | None
    verification_history: list[dict]
    faithfulness_score: float | None
    faithfulness_error: str
    recovery_attempt: int
    recovery_budget: int
    progress_fingerprint: str
    # Build-Verify-Fix self-improvement memory
    stall_count: int
    bvf_lessons: list[str]
    bvf_history: list[dict[str, Any]]
    fix_strategy: str
    researched_gaps: list[str]
    publication: dict | None
    # Human-in-the-loop rejection: the reviewer's free-text feedback and where the
    # feedback-router classifier decided it should be addressed (see ApprovalNodes).
    human_feedback: str
    feedback_route: str
    reviewer: str  # actor who answered the approval interrupt; approvals are recorded for them
    # writer node: the verified report the presented `draft` was formatted from (publication is
    # bound to it), how many reviewer formatting requests were applied, and whether the last one
    # left the report identical.
    verified_draft: str
    format_revision: int
    format_unchanged: bool
    input_turns: list[str]
    scope_questions: list[str]
    checkpoint_metadata: dict[str, Any]
    current_action: str
    activity: list[dict[str, Any]]
    last_error: str


__all__ = ["GraphState", "ResearchGraphState"]
