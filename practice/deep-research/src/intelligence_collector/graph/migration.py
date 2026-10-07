"""Conservative, additive checkpoint migrations. Never invent approvals or evidence."""

from intelligence_collector.contracts import VerificationIssue


def normalize_checkpoint(state: dict) -> dict:
    version = state.get("checkpoint_schema_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError(f"Unsupported checkpoint schema version: {version}")
    result = dict(state)
    result["checkpoint_schema_version"] = 1
    for name, value in {
        "research_gap_attempt": 0,
        "research_gap_budget": 1,
        "verifier_attempt": 0,
        "reviewer_requests": [],
        "research_gaps": [],
        # Supervisor–searcher rounds (spec 002).
        "research_round": 1,
        "gap_stall_count": 0,
        "follow_up_count": 0,
        "gap_analysis": None,
        "claim_support_score": None,
        "claim_evaluation_ref": "",
    }.items():
        result.setdefault(name, value)
    # Preserve consumed legacy allowance; never grant another automatic attempt on restart.
    result["research_gap_budget"] = min(result["research_gap_budget"], 1)
    progress = result.get("progress_assessment") or {}
    legacy_status = progress.get("status") or (
        "READY"
        if progress.get("ready_for_synthesis") is True
        else "INSUFFICIENT"
        if progress.get("ready_for_synthesis") is False
        else "UNKNOWN"
    )
    result.setdefault("research_status", legacy_status)
    result.setdefault("research_limitations", list(progress.get("blockers") or []))
    result.setdefault("result_incomplete", legacy_status in {"PARTIAL", "INSUFFICIENT"})
    result.pop("plan_repair_instruction", None)
    # The research evidence judge was removed: a run stopped on its failure goes to review.
    result.pop("research_verifier_attempt", None)
    if result.get("verification_issue") == "research_verifier_failure":
        result["verification_issue"] = VerificationIssue.RESEARCH_FAILURE.value
    if "todos" in result:
        checklist = _checklist(result.get("scoped_request"))
        result["todos"] = [
            _with_evidence_points(
                {"research_attempt": 1, "research_instruction": "", **todo}, checklist
            )
            for todo in result["todos"]
        ]
    return result


def _checklist(brief) -> list[str]:
    """The brief's required answers, or its question when it lists none (``ResearchBrief``)."""
    brief = brief or {}
    question = str(brief.get("research_question") or "").strip()
    return list(brief.get("required_answers") or []) or ([question] if question else [])


def _with_evidence_points(todo: dict, checklist: list[str]) -> dict:
    """A todo planned before evidence points existed gets one point, linked to every required
    answer so that a missing answer still counts as essential. The stored plan and its hash are
    not touched."""
    if todo.get("evidence_points"):
        return todo
    text = str(todo.get("evidence_needed") or todo.get("research_question") or "").strip()
    if not text:
        return todo
    point = {"point_id": "E1", "text": text[:300], "required_answers": list(checklist)}
    return {**todo, "evidence_points": [point]}
