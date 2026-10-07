"""Guardrail agent. Two layers:

1. ``check_input`` (deterministic, called by the runner before the graph): normalizes the text and
   rejects empty, oversized or credential-bearing input, so it is never stored or sent to a model.
2. ``GuardrailAgent.check`` (LLM, first graph node): blocks unsafe requests and prompt injection.
   A blocked message is removed from the thread so it never reaches the brief.
"""

from __future__ import annotations

import re
import unicodedata

from langchain_core.messages import AIMessage, RemoveMessage

from deep_research.prompts import GUARDRAIL_PROMPT
from deep_research.state import ResearchState, SafetyVerdict

MAX_INPUT_CHARS = 10_000

_INVISIBLE = re.compile(r"[​-‍⁠﻿\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")
_CREDENTIAL = re.compile(
    r"\b(?:api[_-]?key|secret|password|token|authorization)\s*[:=]\s*\S{8,}"
    r"|\bbearer\s+[a-z0-9._-]{8,}|\b(?:sk|tvly|gsk|pk-lf|sk-lf)-[a-z0-9_-]{8,}|\bAIza[0-9a-z_-]{20,}",
    re.I,
)


class GuardrailViolation(ValueError):
    """The input was rejected before reaching the graph; the message is safe to show."""


def check_input(text: str) -> str:
    """Return the normalized text, or raise ``GuardrailViolation``."""
    text = _INVISIBLE.sub("", unicodedata.normalize("NFKC", text or "")).strip()
    if not text:
        raise GuardrailViolation("The message is empty.")
    if len(text) > MAX_INPUT_CHARS:
        raise GuardrailViolation(f"The message is longer than {MAX_INPUT_CHARS:,} characters.")
    if _CREDENTIAL.search(text):
        raise GuardrailViolation("Remove the credential from your message; keys belong in .env.")
    return text


class GuardrailAgent:
    def __init__(self, llm):
        self.llm = llm.structured(SafetyVerdict)

    def check(self, state: ResearchState):
        message = state["messages"][-1]
        verdict = self.llm.invoke(GUARDRAIL_PROMPT.format(message=message.text))
        if verdict.allowed:
            return {"blocked": False}
        notice = AIMessage(f"I can't research that: {verdict.reason or 'the request was blocked.'}")
        return {"blocked": True, "messages": [RemoveMessage(id=message.id), notice]}
