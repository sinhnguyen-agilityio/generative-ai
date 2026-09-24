from __future__ import annotations

from dataclasses import dataclass

from intelligence_collector.contracts import IssueType, ResearchGraphState, Stage


@dataclass(frozen=True)
class RecoveryDecision:
    stage: Stage
    attempt: int
    exhausted: bool
    reason: str


def choose_repair(state: ResearchGraphState, issue: IssueType, fingerprint: str) -> RecoveryDecision:
    next_attempt = state.recovery_attempt + 1
    exhausted = next_attempt > state.recovery_budget or fingerprint == state.progress_fingerprint
    if exhausted:
        return RecoveryDecision(Stage.NEEDS_REVIEW, next_attempt, True, "Recovery budget exhausted or no progress")
    target = {
        IssueType.DRAFT_DEFECT: Stage.SYNTHESIZING,
        IssueType.EVIDENCE_GAP: Stage.RESEARCHING,
        IssueType.PLAN_GAP: Stage.PLANNING,
    }[issue]
    return RecoveryDecision(target, next_attempt, False, issue.value)

