from intelligence_collector.agents.scope import missing_scope_fields, narrowing_suggestions, scope_ready

def test_scope_requires_all_fields_and_suggests_narrowing():
    scope = {"objective": "Compare databases"}
    assert missing_scope_fields(scope) == ["research_boundaries", "constraints", "success_criteria", "deliverable_format"]
    assert narrowing_suggestions(scope)
    assert not scope_ready(scope)

def test_scope_answers_accumulate_until_ready():
    scope = {"objective": "Compare databases", "research_boundaries": "Postgres and SQLite in local apps", "constraints": "local only", "success_criteria": "tradeoffs supported by sources", "deliverable_format": "technical report"}
    assert scope_ready(scope)
