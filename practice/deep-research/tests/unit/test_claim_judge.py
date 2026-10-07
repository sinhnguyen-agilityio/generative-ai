"""Claim-level faithfulness gate (spec 002, FR-022 to FR-028; contracts/claim-evaluation.md)."""

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from intelligence_collector.agents.claim_judge import ClaimJudge, claim_check, segment
from intelligence_collector.models.router import ModelRouter
from tests.fixtures.fake_models import FakeClaimModel, judge_claims

FIXTURE = json.loads(
    (Path(__file__).parents[2] / "evals/supervisor_searcher/claims.json").read_text(encoding="utf-8")
)
SNAPSHOT = FIXTURE["snapshot"]
VERDICTS = {
    "mis_cited": ("unsupported", True, {"S2", "S3"}),
    "partly_supported": ("partly_supported", True, set()),
    "unsupported": ("unsupported", True, {"S1"}),
    "contradicted": ("contradicted", True, set()),
    "uncited": ("supported", True, set()),
}


def report(sentences):
    body = "\n".join(sentences)
    return f"# Report\n\n## Scope\nThis report answers the brief.\n\n## Findings\n{body}\n\n## Sources\n- [S1] x - https://a\n"


def judge(model=None):
    scripted = {
        s["sentence"]: VERDICTS[s["label"]] for s in FIXTURE["sentences"] if s["label"] in VERDICTS
    }
    return ClaimJudge(ModelRouter(model or FakeClaimModel(scripted)))


def test_segmentation_skips_framing_sections_analysis_and_table_headers():
    text = """# Title

## Scope
This report compares two databases.

## Findings
SQLite allows one writer at a time [S1]. Python 3.13 adds a JIT [S2, S3]; e.g. this is new.
- SQLite is in the public domain [S3].
Assessment: SQLite fits single-writer apps.

| Item | Value |
| --- | --- |
| Size | 281 TB [S2] |

## Limitations
- Nothing about Windows.
"""
    sentences = segment(text)
    assert [s.text for s in sentences] == [
        "SQLite allows one writer at a time [S1].",
        "Python 3.13 adds a JIT [S2, S3]; e.g. this is new.",
        "SQLite is in the public domain [S3].",
        "Size | 281 TB [S2]",
    ]
    assert sentences[1].cited == ("S2", "S3") and sentences[0].section == "Findings"
    assert [s.claim_id for s in sentences] == ["K1", "K2", "K3", "K4"]


def test_every_planted_error_fails_with_its_code_and_supported_sentences_pass():
    sentences = [s for s in FIXTURE["sentences"]]
    evaluation = judge().evaluate(
        report([s["sentence"] for s in sentences]), SNAPSHOT, draft_version=1, artifact_hash="h"
    )
    by_sentence = {c.sentence: c for c in evaluation.claims}
    for item in sentences:
        if item["label"] == "analysis":
            assert item["sentence"] not in by_sentence  # labeled analysis is never judged
            continue
        claim = by_sentence[item["sentence"]]
        if item["failure"] is None:
            assert claim.failures == [], item
        else:
            assert any(f.startswith(item["failure"]) for f in claim.failures), (item, claim.failures)
    mis_cited = by_sentence["SQLite allows only one writer at a time [S1][S3]."]
    assert "source_supports_nothing:S3" in mis_cited.failures
    assert not evaluation.passed
    factual = [c for c in evaluation.claims if c.states_fact]
    assert evaluation.supported_share == pytest.approx(2 / len(factual))


def test_a_sentence_combining_two_sources_passes_when_each_supports_a_part():
    text = report(["SQLite databases can grow to 281 TB and are in the public domain [S2][S3]."])
    evaluation = judge().evaluate(text, SNAPSHOT, draft_version=1, artifact_hash="h")
    assert evaluation.passed and evaluation.supported_share == 1.0


def test_framing_sentences_are_recorded_but_not_scored():
    text = report(["This section compares the two engines.", "SQLite allows only one writer at a time [S1]."])
    evaluation = ClaimJudge(ModelRouter(FakeClaimModel())).evaluate(
        text, SNAPSHOT, draft_version=1, artifact_hash="h"
    )
    framing = evaluation.claims[0]
    assert framing.states_fact is False and framing.failures == []
    assert evaluation.passed and evaluation.supported_share == 1.0


def test_the_judge_only_sees_the_cited_sources_snapshot_text():
    seen = []

    class Recording(FakeClaimModel):
        def invoke(self, messages, config=None):
            seen.append(str(messages[-1].content))
            return super().invoke(messages, config)

    ClaimJudge(ModelRouter(Recording())).evaluate(
        report(["SQLite allows only one writer at a time [S1]."]), SNAPSHOT,
        draft_version=1, artifact_hash="h",
    )
    assert "only be one writer" in seen[0]
    assert "281 TB" not in seen[0] and "public domain" not in seen[0]  # S2, S3 not cited


def test_missing_verdicts_get_one_supplemental_call_then_error():
    class Omits:
        model_name = "omits"

        def __init__(self, fill_on_retry):
            self.calls = 0
            self.fill_on_retry = fill_on_retry

        def invoke(self, messages, config=None):
            self.calls += 1
            answer = judge_claims(messages)
            if self.calls == 1 or not self.fill_on_retry:
                answer["claims"] = [c for c in answer["claims"] if c["claim_id"] != "K2"]
            return AIMessage(content=json.dumps(answer))

    text = report(["SQLite allows only one writer at a time [S1].",
                   "The maximum database size is 281 TB [S2]."])
    recovered = Omits(fill_on_retry=True)
    evaluation = ClaimJudge(ModelRouter(recovered)).evaluate(text, SNAPSHOT, draft_version=1, artifact_hash="h")
    assert recovered.calls == 2 and len(evaluation.claims) == 2
    with pytest.raises(ValueError, match="no verdict for K2"):
        ClaimJudge(ModelRouter(Omits(fill_on_retry=False))).evaluate(
            text, SNAPSHOT, draft_version=1, artifact_hash="h"
        )


def test_claim_check_routes_failures_to_the_faithfulness_gate():
    evaluation = judge().evaluate(
        report(["SQLite allows only one writer at a time [S2].", "SQLite allows only one writer at a time [S1]."]),
        SNAPSHOT, draft_version=1, artifact_hash="h",
    )
    check = claim_check(evaluation)
    assert check.gate == "faithfulness" and not check.passed
    assert check.missing_points == ["SQLite allows only one writer at a time [S2]."]
    assert check.affected_ids == ["claim:K1"]
