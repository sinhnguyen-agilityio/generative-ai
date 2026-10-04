"""Conservative, additive checkpoint migrations. Never invent approvals or evidence."""


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
        "research_verifier_attempt": 0,
        "reviewer_requests": [],
        "research_gaps": [],
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
    if "todos" in result:
        result["todos"] = [
            {"research_attempt": 1, "research_instruction": "", **todo} for todo in result["todos"]
        ]
    return result
