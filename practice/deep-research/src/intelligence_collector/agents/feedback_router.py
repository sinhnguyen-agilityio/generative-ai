"""Classifies a human reviewer's chat reply (the HITL approval gate, see
``graph/nodes.py::ApprovalNodes``) as an approval, a formatting-only change or a scope change.
See ``FEEDBACK_ROUTER_PROMPT`` for the exact rule the model applies.
"""

from __future__ import annotations

from intelligence_collector.agents.prompts import FEEDBACK_ROUTER_TEMPLATE
from intelligence_collector.contracts import FeedbackDecision

# The verdict depends on what the reviewer asked for, not on the whole report; a short excerpt
# gives enough context and keeps the request small enough to stay under provider token limits.
MAX_REPORT_EXCERPT_CHARS = 3000


class FeedbackRouter:
    """Reads the reviewer's chat reply with an LLM (role ``feedback_router``) and returns a
    ``FeedbackDecision``.
    """

    def __init__(self, router):
        self.router = router

    def route(
        self,
        feedback: str,
        report: str,
        *,
        run_id: str = "",
        thread_id: str = "",
    ) -> FeedbackDecision:
        """Return `route="approve"` (publish), `"format"` (re-present the existing text) or
        `"scope"` (re-scope)."""
        messages = FEEDBACK_ROUTER_TEMPLATE.format_messages(
            feedback=feedback, report=report[:MAX_REPORT_EXCERPT_CHARS],
        )
        return self.router.invoke_messages(
            "feedback_router",
            messages,
            FeedbackDecision,
            run_id=run_id,
            thread_id=thread_id,
        )
