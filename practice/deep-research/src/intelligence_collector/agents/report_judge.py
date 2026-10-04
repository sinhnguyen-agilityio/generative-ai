"""Coverage gate: one reasoning-tier LLM call, with no search tools, that decides whether the
report answers every required item of the brief and follows the output requirements.

Coverage is the first of the three verification gates; citation resolution
(``source_verification.citation_errors``) and RAGAS Faithfulness are the other two. Each
criterion carries a stable id (``[C1]``), so the verdict is matched by id rather than by the
model copying long criterion text verbatim. Duplicates keep the first verdict; criteria the
model omits get one supplemental call; anything still missing is a technical ERROR, never a
FAIL.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from intelligence_collector.agents.prompts import REPORT_JUDGE_TEMPLATE
from intelligence_collector.contracts import (
    ReportJudgeCriterion,
    ReportJudgeVerdict,
    ResearchBrief,
    VerificationCheck,
    VerificationResult,
)

_ID = re.compile(r"^\s*\[?(C\d+)\]?[\s:.)-]*", re.IGNORECASE)


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "(none)"


@dataclass(frozen=True)
class CoverageCriterion:
    """One coverage criterion: a required answer (``kind="answer"``) or an output constraint."""

    criterion_id: str
    text: str
    kind: str  # "answer" | "output"

    @property
    def label(self) -> str:
        return f"[{self.criterion_id}] {self.text}"


def build_coverage_criteria(brief: ResearchBrief, guidelines: list[str] = (),
                            reviewer_requests: list[str] = ()) -> list[CoverageCriterion]:
    """Required answers first, then explicit output requirements, operator guidelines and
    reviewer requests."""
    texts = [("answer", f"Answers: {item}") for item in brief.checklist]
    texts += [("output", f"Output requirement: {item}") for item in brief.output_requirements]
    texts += [("output", item) for item in guidelines]
    texts += [("output", f"Satisfies this reviewer request: {item}") for item in reviewer_requests]
    return [CoverageCriterion(f"C{index}", text, kind)
            for index, (kind, text) in enumerate(texts, start=1)]


class ReportJudge:
    """The coverage gate (role ``verify``)."""

    def __init__(self, router):
        self.router = router

    def _ask(self, brief, criteria, report, evidence, reviewer_requests, run_id, thread_id):
        messages = REPORT_JUDGE_TEMPLATE.format_messages(
            original_question=brief.original_question or brief.research_question,
            brief=brief.research_question,
            criteria=_bullets([c.label for c in criteria]),
            reviewer_note=(
                f"Reviewer requests (applied to the report):\n{_bullets(list(reviewer_requests))}\n\n"
                if reviewer_requests else ""
            ),
            evidence=evidence,
            report=report,
        )
        return self.router.invoke_messages(
            "verify", messages, ReportJudgeVerdict, run_id=run_id, thread_id=thread_id,
        )

    @staticmethod
    def _match(verdict: ReportJudgeVerdict, criteria: list[CoverageCriterion],
               found: dict[str, ReportJudgeCriterion]) -> None:
        by_text = {c.text.strip().casefold(): c.criterion_id for c in criteria}
        wanted = {c.criterion_id for c in criteria}
        for item in verdict.results:
            match = _ID.match(item.criterion)
            criterion_id = match.group(1).upper() if match else by_text.get(
                item.criterion.strip().casefold())
            if criterion_id in wanted and criterion_id not in found:
                found[criterion_id] = item

    def assess(
        self,
        brief: ResearchBrief,
        criteria: list[CoverageCriterion],
        report: str,
        evidence: str,
        *,
        reviewer_requests: list[str] = (),
        run_id: str = "",
        thread_id: str = "",
    ) -> list[tuple[CoverageCriterion, ReportJudgeCriterion]]:
        """Judge every criterion; raises ``ValueError`` (a technical ERROR) when the model cannot
        produce a verdict for each criterion after one supplemental call."""
        found: dict[str, ReportJudgeCriterion] = {}
        self._match(self._ask(brief, criteria, report, evidence, reviewer_requests,
                              run_id, thread_id), criteria, found)
        missing = [c for c in criteria if c.criterion_id not in found]
        if missing:
            self._match(self._ask(brief, missing, report, evidence, reviewer_requests,
                                  run_id, thread_id), missing, found)
        missing = [c.criterion_id for c in criteria if c.criterion_id not in found]
        if missing:
            raise ValueError("Coverage judge returned no verdict for " + ", ".join(missing))
        return [(c, found[c.criterion_id]) for c in criteria]


def _affected_id(text: str) -> str:
    return f"criterion:{hashlib.sha1(text.strip().casefold().encode()).hexdigest()[:8]}"


def coverage_checks(pairs) -> list[VerificationCheck]:
    """Coverage verdicts as gate checks. A required answer judged "presentation" stays a
    draft defect: an unanswered question is never only a layout problem."""
    checks = []
    for criterion, verdict in pairs:
        issue = verdict.issue_type
        if criterion.kind == "answer" and issue == "presentation":
            issue = "draft_defect"
        item = criterion.text.removeprefix("Answers: ")
        checks.append(VerificationCheck(
            name=f"coverage_{criterion.criterion_id}",
            passed=verdict.met,
            gate="coverage",
            issue_type=issue if not verdict.met else "draft_defect",
            item=item,
            missing_points=[] if verdict.met else list(verdict.missing_points),
            details=[] if verdict.met else [f"{criterion.text} — {verdict.justification}" + (
                " Missing: " + "; ".join(verdict.missing_points) if verdict.missing_points else "")],
            affected_ids=[] if verdict.met else [_affected_id(criterion.text)],
        ))
    return checks


def verification_result(checks: list[VerificationCheck], artifact_hash: str) -> VerificationResult:
    """One verification record for the exact draft, from every gate's checks."""
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
