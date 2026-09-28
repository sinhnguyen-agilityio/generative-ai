"""Classifies a human reviewer's rejection feedback (the HITL approval gate, see
``graph/nodes.py::ApprovalNodes``) as needing either a report-writer revision or a planner
replan. See ``FEEDBACK_ROUTER_PROMPT`` for the exact rule the model applies.
"""

from __future__ import annotations

from intelligence_collector.agents.prompts import (
    FEEDBACK_ROUTER_PROMPT_VERSION,
    FEEDBACK_ROUTER_TEMPLATE,
)
from intelligence_collector.contracts import FeedbackDecision


class FeedbackRouter:
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
        messages = FEEDBACK_ROUTER_TEMPLATE.format_messages(feedback=feedback, report=report)
        return self.router.invoke_messages(
            "feedback_router",
            messages,
            FeedbackDecision,
            prompt_version=FEEDBACK_ROUTER_PROMPT_VERSION,
            run_id=run_id,
            thread_id=thread_id,
        )
