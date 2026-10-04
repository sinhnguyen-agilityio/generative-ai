"""Stage-only placeholders used when no agents are injected (wiring tests)."""

from __future__ import annotations

from intelligence_collector.graph.state import GraphState


def stage_only(name: str):
    """Placeholder node that only records its own name as the stage."""

    def advance(state: GraphState) -> GraphState:
        return {**state, "stage": name}

    return advance
