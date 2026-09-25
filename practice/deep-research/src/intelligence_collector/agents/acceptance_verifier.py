"""Acceptance-criteria verification: the goal-level "does this report achieve the brief?" check.

Runs only after the report already passed the deterministic ``source_verification`` script — it
judges content and completeness against the planner's acceptance criteria, not citation syntax or
grounding (that is already enforced without any LLM call). A failed criterion is folded into the
same ``VerificationResult`` shape as a source-verification failure, so it flows through the
existing BVF fix/reflect loop unchanged.
"""

from __future__ import annotations

from intelligence_collector.agents.prompts import ACCEPTANCE_PROMPT_VERSION, ACCEPTANCE_TEMPLATE
from intelligence_collector.contracts import (
    AcceptanceVerdict,
    IssueType,
    ResearchBrief,
    VerificationCheck,
    VerificationResult,
)

ACCEPTANCE_CHECK_NAME = "acceptance_criteria"


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "(none)"


class AcceptanceVerifier:
    def __init__(self, router):
        self.router = router

    def verify(
        self,
        brief: ResearchBrief,
        criteria: list[str],
        report: str,
        *,
        run_id: str = "",
        thread_id: str = "",
    ) -> AcceptanceVerdict:
        messages = ACCEPTANCE_TEMPLATE.format_messages(
            brief=brief.research_question,
            criteria=_bullets(criteria),
            report=report,
        )
        return self.router.invoke_messages(
            "verify",
            messages,
            AcceptanceVerdict,
            prompt_version=ACCEPTANCE_PROMPT_VERSION,
            run_id=run_id,
            thread_id=thread_id,
        )


def merge_acceptance_verdict(
    result: VerificationResult, verdict: AcceptanceVerdict
) -> VerificationResult:
    """Fold a failed acceptance verdict into the source-verification result unchanged otherwise.

    Only called when ``result.passed`` is already true, so this can only turn a pass into a
    failure, never mask one of the deterministic checks.
    """
    unmet = [item for item in verdict.results if not item.met]
    if not unmet:
        return result
    check = VerificationCheck(
        name=ACCEPTANCE_CHECK_NAME,
        passed=False,
        issue_type=IssueType.DRAFT_DEFECT,
        details=[f'Not met: "{item.criterion}" — {item.justification}' for item in unmet],
        affected_ids=[f"acceptance:{index}" for index in range(len(unmet))],
    )
    repair_details = "\n".join(
        filter(None, [result.repair_details, *(f"[{check.name}] {detail}" for detail in check.details)])
    )
    return result.model_copy(update={
        "passed": False,
        "checks": [*result.checks, check.name],
        "issue_type": IssueType.DRAFT_DEFECT,
        "affected_ids": [*result.affected_ids, *check.affected_ids],
        "repair_details": repair_details,
        "check_results": [*result.check_results, check],
    })
