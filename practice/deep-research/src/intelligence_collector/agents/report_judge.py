"""LLM-as-judge report verification: the sole "Verify" step of Build-Verify-Fix.

Replaces the deterministic ``source_verification`` script as the Verify gate. One reasoning-tier
LLM call, with no search or scrape tools, re-reads the writer's report against the evidence the
research step already gathered and checks two things:

* fact check: does the report invent, distort or contradict anything relative to that evidence
  (i.e. did the Writer hallucinate while synthesizing)? It does NOT re-verify the evidence itself
  against the internet -- that was the researchers' job;
* guidelines: does it follow the operator-configured report guidelines
  (see ``config.DEFAULT_REPORT_GUIDELINES``)?

Each criterion verdict carries its own ``issue_type`` (see ``ReportJudgeCriterion``), so a report
that fails because the evidence is genuinely insufficient still routes to targeted research
instead of a pointless rewrite -- the same Build-Verify-Fix routing the deterministic script drove.
"""

from __future__ import annotations

import hashlib

from intelligence_collector.agents.prompts import (
    REPORT_JUDGE_TEMPLATE,
)
from intelligence_collector.contracts import (
    IssueType,
    ReportJudgeVerdict,
    ResearchBrief,
    VerificationCheck,
    VerificationResult,
)
from intelligence_collector.source_verification import EvidenceGap, SourceCatalog


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "(none)"


FACT_CHECK_CRITERION = (
    "Contains no factual inaccuracies, hallucinations, or internal inconsistencies. Read the "
    "whole report as one continuous piece -- introduction through conclusion -- and check every "
    "claim for truthfulness against the evidence above, not merely whether it carries a "
    "citation; if you find any inaccuracy, name each one specifically."
)


def build_report_judge_criteria(guidelines: list[str] = ()) -> list[str]:
    """The checklist handed to the judge: the fixed fact-check criterion, then the
    operator-configured guideline checklist (formatting/language rules)."""
    return [FACT_CHECK_CRITERION, *guidelines]


def _known_gaps(catalog: SourceCatalog, researched_gaps: list[str]) -> str:
    researched = set(researched_gaps)
    gaps: list[EvidenceGap] = [gap for gap in catalog.gaps if gap.gap_id not in researched]
    return _bullets([f"{gap.question}: {gap.text}" for gap in gaps])


class ReportJudge:
    def __init__(self, router):
        self.router = router

    def verify(
        self,
        brief: ResearchBrief,
        criteria: list[str],
        report: str,
        evidence: str,
        catalog: SourceCatalog,
        *,
        researched_gaps: list[str] | None = None,
        reviewer_request: str = "",
        run_id: str = "",
        thread_id: str = "",
    ) -> ReportJudgeVerdict:
        """Judge `report` against `criteria`. `reviewer_request` is a human's formatting request
        already applied to the report; it overrides conflicting formatting guidelines."""
        messages = REPORT_JUDGE_TEMPLATE.format_messages(
            brief=brief.research_question,
            criteria=_bullets(criteria),
            reviewer_note=(
                f"Reviewer formatting request (already applied to the report):\n{reviewer_request}\n\n"
                if reviewer_request else ""
            ),
            evidence=evidence,
            gaps=_known_gaps(catalog, researched_gaps or []),
            report=report,
        )
        return self.router.invoke_messages(
            "verify",
            messages,
            ReportJudgeVerdict,
            run_id=run_id,
            thread_id=thread_id,
        )


def judge_verdict_to_result(verdict: ReportJudgeVerdict, artifact_hash: str) -> VerificationResult:
    """Build a fresh ``VerificationResult`` straight from the judge's verdict — there is no
    deterministic result to merge onto; this verdict is the entire Verify outcome.

    ``affected_ids`` is a hash of the criterion's own text, not its position in the list — the
    same criterion failing on a later attempt (even with different justification wording) must
    produce the same fingerprint (see ``failure_fingerprint``), so the BVF loop's no-progress
    stall detection still works; a different criterion failing must produce a different one."""
    checks = [
        VerificationCheck(
            name=f"criterion_{index}",
            passed=item.met,
            issue_type=None if item.met else item.issue_type,
            details=[] if item.met else [f"{item.criterion} — {item.justification}"],
            affected_ids=[] if item.met else [
                f"criterion:{hashlib.sha1(item.criterion.strip().casefold().encode()).hexdigest()[:8]}"
            ],
        )
        for index, item in enumerate(verdict.results, start=1)
    ]
    failed = [check for check in checks if not check.passed]
    issue = None
    if failed:
        issue = (
            IssueType.EVIDENCE_GAP
            if any(check.issue_type == IssueType.EVIDENCE_GAP for check in failed)
            else IssueType.DRAFT_DEFECT
        )
    return VerificationResult(
        artifact_hash=artifact_hash,
        passed=not failed,
        checks=[check.name for check in checks],
        issue_type=issue,
        affected_ids=list(dict.fromkeys(a for check in failed for a in check.affected_ids)),
        repair_details="\n".join(
            f"[{check.name}] {detail}" for check in failed for detail in check.details
        ),
        check_results=checks,
    )
