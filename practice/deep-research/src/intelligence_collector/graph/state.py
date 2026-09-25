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
    recovery_attempt: int
    recovery_budget: int
    progress_fingerprint: str
    publication: dict | None
    input_turns: list[str]
    scope_questions: list[str]
    checkpoint_metadata: dict[str, Any]
    current_action: str
    activity: list[dict[str, Any]]
    last_error: str


__all__ = ["GraphState", "ResearchGraphState"]
