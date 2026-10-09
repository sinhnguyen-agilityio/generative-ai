"""Guardrail agent. Two layers:

1. ``check_input`` (deterministic, called by the runner before the graph): normalizes the text and
   rejects empty, oversized or credential-bearing input, so it is never stored or sent to a model.
2. ``GuardrailAgent.check`` (LLM, first graph node): blocks unsafe requests and prompt injection.
   A blocked message is removed from the thread so it never reaches the brief.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, RemoveMessage

from deep_research.prompts import GUARDRAIL_PROMPT
from deep_research.schemas.safety_verdict import SafetyVerdict
from deep_research.states.research_state import ResearchState


class GuardrailAgent:
    def __init__(self, llm):
        """Configure the structured safety classifier.

        Args:
            llm: Model tier producing a SafetyVerdict.
        """
        self.llm = llm.structured(SafetyVerdict)

    def check(self, state: ResearchState):
        """Classify the latest request and remove it from the thread when blocked.

        Args:
            state: Conversation containing the latest user message.

        Returns:
            Blocked flag and, for rejected requests, message removal plus an assistant notice.
        """
        message = state["messages"][-1]
        verdict = self.llm.invoke(GUARDRAIL_PROMPT.format(message=message.text))
        if verdict.allowed:
            return {"blocked": False}
        notice = AIMessage(f"I can't research that: {verdict.reason or 'the request was blocked.'}")
        return {"blocked": True, "messages": [RemoveMessage(id=message.id), notice]}
