"""Scope Agent: the content-safety guardrail, the clarify-or-proceed decision, and the research
brief.
"""

from __future__ import annotations
from datetime import date
from intelligence_collector.contracts import (
    ResearchBrief,
    ScopeAssessment,
)
from intelligence_collector.agents.prompts import (
    BRIEF_TEMPLATE,
    GUARDRAIL_TEMPLATE,
    SCOPE_TEMPLATE,
)
from intelligence_collector.guardrails import ContentSafetyVerdict, GuardrailViolation
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from intelligence_collector.storage.vfs import VFS

class ScopeAgent:
    """Turns the user's chat into a research brief. `router` runs the scope and brief calls;
    `guardrail_router` (default: the same router) runs the content-safety check.
    """

    def __init__(self, router, guardrail_router=None):
        self.router = router
        self.guardrail_router = guardrail_router or router

    @staticmethod
    def _messages(prior_turns: list[str | BaseMessage] | None) -> list[BaseMessage]:
        if not prior_turns:
            return []
        if all(isinstance(turn, BaseMessage) for turn in prior_turns):
            return list(prior_turns)  # type: ignore[arg-type]
        return [
            HumanMessage(content=turn) if i % 2 == 0 else AIMessage(content=turn)
            for i, turn in enumerate(prior_turns)
        ]

    def check_content_safety(
        self,
        request: str,
        *,
        run_id: str = "",
        thread_id: str = "",
    ) -> ContentSafetyVerdict:
        """Pre-model guardrail: classify the user's last message before it reaches the scope
        model. Runs on the cheap standard-tier model so a blocked message never pays for the
        deep-reasoning scope call."""
        messages = GUARDRAIL_TEMPLATE.format_messages(message=request)
        return self.guardrail_router.invoke_messages(
            "guardrail",
            messages,
            ContentSafetyVerdict,
            run_id=run_id,
            thread_id=thread_id,
        )

    def clarify_with_user(
        self,
        request: str,
        prior_turns: list[str | BaseMessage] | None = None,
        *,
        run_id: str = "",
        thread_id: str = "",
    ) -> ScopeAssessment:
        """Check the latest message with the guardrail, then decide from the whole conversation
        whether to ask one clarifying question or confirm the scope. Raises
        ``GuardrailViolation`` for a blocked request.
        """
        verdict = self.check_content_safety(request, run_id=run_id, thread_id=thread_id)
        if not verdict.allowed:
            raise GuardrailViolation(verdict.reason or "This request cannot be researched.")
        conversation = self._messages(prior_turns) + [HumanMessage(content=request)]
        messages = SCOPE_TEMPLATE.format_messages(
            conversation=conversation,
            date=date.today().isoformat(),
        )
        return self.router.invoke_messages(
            "scope",
            messages,
            ScopeAssessment,
            run_id=run_id,
            thread_id=thread_id,
        )

    def write_research_brief(
        self,
        request: str,
        assessment: ScopeAssessment,
        prior_turns: list[str | BaseMessage] | None = None,
        *,
        run_id: str = "",
        thread_id: str = "",
        vfs: VFS | None = None,
    ) -> ResearchBrief:
        """Write the ``ResearchBrief`` from the conversation once the scope is clear, saving it to
        the VFS when `vfs` is given. Raises ``ValueError`` if `assessment` still needs
        clarification.
        """
        if assessment.need_clarification:
            raise ValueError("write_research_brief requires a clear scope decision")
        conversation = self._messages(prior_turns)
        messages = BRIEF_TEMPLATE.format_messages(
            conversation=conversation,
            date=date.today().isoformat(),
        )
        brief = self.router.invoke_messages(
            "scope",
            messages,
            ResearchBrief,
            run_id=run_id,
            thread_id=thread_id,
        )
        if vfs and run_id:
            vfs.write_json(
                run_id,
                "scope/brief_summary.v1.json",
                brief.model_dump(mode="json"),
                "brief_summary",
            )
        return brief

    def assess(
        self,
        request: str,
        prior_turns: list[str | BaseMessage] | None = None,
        *,
        run_id: str = "",
        thread_id: str = "",
        vfs: VFS | None = None,
    ) -> ScopeAssessment | ResearchBrief:
        """Clarify, and when the scope is clear also write the brief. Returns the assessment or the
        brief.
        """
        assessment = self.clarify_with_user(
            request, prior_turns, run_id=run_id, thread_id=thread_id
        )
        if not assessment.need_clarification:
            return self.write_research_brief(
                request,
                assessment,
                prior_turns,
                run_id=run_id,
                thread_id=thread_id,
                vfs=vfs,
            )
        return assessment
