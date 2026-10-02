"""LLM-as-judge report verification: the sole "Verify" step of Build-Verify-Fix.

Replaces the deterministic ``source_verification`` script as the Verify gate. One reasoning-tier
LLM call, with no search or scrape tools, re-reads the writer's report against the evidence the
research step already gathered and checks two things:

* fact check: does the report invent, distort or contradict anything relative to that evidence
  (i.e. did the Writer hallucinate while synthesizing)? It does NOT re-verify the evidence itself
  against the internet -- that was the researchers' job;
* guidelines: does it follow the operator-configured report guidelines
  (see ``config.DEFAULT_REPORT_GUIDELINES``)?

The judge classifies failures as draft defects, missing evidence, or missing plan coverage;
the graph selects the bounded repair loop for that classification.
"""

from __future__ import annotations

import hashlib
from collections import Counter

from intelligence_collector.agents.prompts import (
    REPORT_JUDGE_TEMPLATE,
)
from intelligence_collector.contracts import (
    ReportJudgeVerdict,
    ResearchBrief,
    VerificationCheck,
    VerificationResult,
)


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


class ReportJudge:
    """The Verify step: one LLM call (role ``verify``) that judges a report against the fact-check
    criterion and the report guidelines.
    """

    def __init__(self, router):
        self.router = router

    def verify(
        self,
        brief: ResearchBrief,
        criteria: list[str],
        report: str,
        evidence: str,
        *,
        reviewer_requests: list[str] = (),
        run_id: str = "",
        thread_id: str = "",
    ) -> ReportJudgeVerdict:
        """Judge `report` against each of `criteria`, using `evidence` (the formatted catalog).
        `reviewer_requests` are the human reviewer's change requests the report applies; they
        override conflicting formatting guidelines, never the evidence rules."""
        messages = REPORT_JUDGE_TEMPLATE.format_messages(
            brief=brief.research_question,
            criteria=_bullets(criteria),
            reviewer_note=(
                f"Reviewer requests (applied to the report):\n{_bullets(list(reviewer_requests))}\n\n"
                if reviewer_requests else ""
            ),
            evidence=evidence,
            report=report,
        )
        messages[0].content += (
            "\nFor each failed criterion classify issue_type: draft_defect when existing evidence "
            "can support a corrected report; evidence_gap when required evidence is missing; "
            "plan_gap when the plan omitted required research. Evidence gaps MUST name "
            "affected_todo_ids from the evidence question ids and give a research_instruction. "
            "Unsupported incidental claims removable from the draft are draft defects."
        )
        verdict = self.router.invoke_messages(
            "verify",
            messages,
            ReportJudgeVerdict,
            run_id=run_id,
            thread_id=thread_id,
        )
        if Counter(item.criterion for item in verdict.results) != Counter(criteria):
            raise ValueError("Report judge must return every requested criterion exactly once")
        return verdict


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
            details=[] if item.met else [f"{item.criterion} — {item.justification}"],
            affected_ids=[] if item.met else [
                f"criterion:{hashlib.sha1(item.criterion.strip().casefold().encode()).hexdigest()[:8]}"
            ],
        )
        for index, item in enumerate(verdict.results, start=1)
    ]
    failed = [check for check in checks if not check.passed]
    return VerificationResult(
        artifact_hash=artifact_hash,
        passed=not failed,
        checks=[check.name for check in checks],
        affected_ids=list(dict.fromkeys(a for check in failed for a in check.affected_ids)),
        repair_details="\n".join(
            f"[{check.name}] {detail}" for check in failed for detail in check.details
        ),
        check_results=checks,
    )
