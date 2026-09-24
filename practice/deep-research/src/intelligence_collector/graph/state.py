from typing import TypedDict
from intelligence_collector.contracts import ResearchGraphState

class GraphState(TypedDict, total=False):
    run_id: str
    thread_id: str
    stage: str
    status: str
    payload: dict
    approval: dict

__all__ = ["GraphState", "ResearchGraphState"]
