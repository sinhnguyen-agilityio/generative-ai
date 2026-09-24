from intelligence_collector.agents.verification import preserve_evidence, repair_draft
def test_repair_preserves_valid_evidence():
    evidence = ["e1"]
    assert preserve_evidence(evidence) == evidence
    assert "fix" in repair_draft("draft", "fix").lower()
