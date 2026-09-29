from __future__ import annotations
import hashlib
from datetime import date
from intelligence_collector.contracts import (
    ResearchBrief,
    ScopeAssessment,
    ScopedRequest,
)
from intelligence_collector.agents.prompts import (
    BRIEF_PROMPT_VERSION,
    BRIEF_TEMPLATE,
    GUARDRAIL_PROMPT_VERSION,
    GUARDRAIL_TEMPLATE,
    PROMPT_VERSION,
    SCOPE_TEMPLATE,
)
from intelligence_collector.guardrails import ContentSafetyVerdict, GuardrailViolation
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from intelligence_collector.storage.vfs import VFS

REQUIRED = ("objective", "research_boundaries", "constraints",
            "success_criteria", "deliverable_format")


def missing_scope_fields(scope: dict[str, str]) -> list[str]:
    return [key for key in REQUIRED if not scope.get(key, "").strip()]


def narrowing_suggestions(scope: dict[str, str]) -> list[str]:
    """Return actionable narrowing prompts for broad or underspecified requests."""
    suggestions = []
    if not scope.get("research_boundaries", "").strip():
        suggestions.append(
            "Limit the topic to a defined audience, time period, or technical scope.")
    if not scope.get("success_criteria", "").strip():
        suggestions.append(
            "State what evidence would make the research answer successful.")
    if not scope.get("deliverable_format", "").strip():
        suggestions.append(
            "Choose one deliverable format, such as a comparison or technical report.")
    if len(scope.get("objective", "")) > 1000:
        suggestions.append(
            "Split the objective into one primary question and explicit exclusions.")
    return suggestions


def scope_ready(scope: dict[str, str]) -> bool:
    return all(scope.get(k, "").strip() for k in REQUIRED)


def build_scope(scope: dict[str, str], vfs: VFS | None = None, run_id: str | None = None) -> ScopedRequest:
    if not scope_ready(scope):
        raise ValueError("scope is incomplete: " +
                         ", ".join(missing_scope_fields(scope)))
    value = "|".join(scope[k].strip() for k in REQUIRED)
    result = ScopedRequest(
        **scope, scope_hash=hashlib.sha256(value.encode()).hexdigest())
    if vfs and run_id:
        vfs.write_json(run_id, "scope/brief_summary.v1.json",
                       result.model_dump(mode="json"), "brief_summary")
    return result


class ScopeAgent:
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
            prompt_version=GUARDRAIL_PROMPT_VERSION,
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
            prompt_version=PROMPT_VERSION,
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
            prompt_version=BRIEF_PROMPT_VERSION,
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
