from typing import TypedDict
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
    artifacts: list[dict]
    last_error: str

__all__ = ["GraphState", "ResearchGraphState"]
