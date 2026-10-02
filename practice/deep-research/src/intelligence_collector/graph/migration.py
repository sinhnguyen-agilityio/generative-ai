"""Conservative, additive checkpoint migrations. Never invent approvals or evidence."""


def normalize_checkpoint(state: dict) -> dict:
    version = state.get("checkpoint_schema_version", 1)
    if type(version) is not int or version != 1:
        raise ValueError(f"Unsupported checkpoint schema version: {version}")
    result = dict(state)
    result["checkpoint_schema_version"] = 1
    for name, value in {"research_gap_attempt": 0, "research_gap_budget": 2,
                        "verifier_attempt": 0, "research_verifier_attempt": 0,
                        "reviewer_requests": [], "research_gaps": []}.items():
        result.setdefault(name, value)
    if "todos" in result:
        result["todos"] = [{"research_attempt": 1, "research_instruction": "", **todo}
                           for todo in result["todos"]]
    return result
