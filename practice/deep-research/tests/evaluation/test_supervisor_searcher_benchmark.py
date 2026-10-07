"""Benchmark metrics on the labeled fixtures with scripted routers (spec 002, T062a)."""

import json
import re

import pytest
from langchain_core.messages import AIMessage

from intelligence_collector.models.router import ModelRouter
from intelligence_collector.observability.supervisor_benchmark import (
    compare,
    load_fixture,
    score_claims,
    score_gap_analysis,
    score_result_evaluation,
)
from tests.fixtures.fake_models import FakeClaimModel, accept_all_results

RESULTS = load_fixture("results.json")
GAPS = load_fixture("gaps.json")
CLAIMS = load_fixture("claims.json")
LABELS = {row["url"]: row["label"] for case in RESULTS["cases"] for row in case["rows"]}
_ROW = re.compile(r"^\[(\d+)\] .*\nURL: (\S+)", re.M)


class LabelEvaluator:
    """Decides every row shown to it exactly as its human label says."""

    model_name = "label-evaluator"

    def invoke(self, messages, config=None):
        rows = _ROW.findall(str(messages[-1].content))
        return AIMessage(content=json.dumps({"rows": [
            {"index": int(n), "decision": "accepted", "covers": ["E1"]}
            if LABELS[url] == "relevant"
            else {"index": int(n), "decision": "rejected", "reason": LABELS[url]}
            for n, url in rows
        ]}))


class AcceptAll:
    model_name = "accept-all"

    def invoke(self, messages, config=None):
        return AIMessage(content=json.dumps(accept_all_results(messages)))


def test_result_evaluation_precision_and_recall():
    perfect = score_result_evaluation(ModelRouter(LabelEvaluator()), RESULTS)
    assert perfect == {"rows": 7, "precision": 1.0, "recall": 1.0}
    # Accepting everything the code lets through keeps recall but admits the off-topic and
    # low-credibility rows; duplicates and empty pages are still rejected by code.
    lax = score_result_evaluation(ModelRouter(AcceptAll()), RESULTS)
    assert lax["recall"] == 1.0 and lax["precision"] == pytest.approx(3 / 5)


class GapSupervisor:
    """Coverage verdicts, then one supervisor turn per script step."""

    model_name = "gap-supervisor"

    def __init__(self, coverage, turns):
        self.coverage, self.turns = coverage, list(turns)

    def bind_tools(self, tools, tool_choice=None):
        return self

    def invoke(self, messages, config=None):
        system = str(messages[0].content)
        if system.startswith("You write lookup queries"):
            return AIMessage(content=json.dumps({"points": []}))
        if "reviewing the end of a research round" in system:
            keys = re.findall(r"^(t[\w-]*:E\d+): ", str(messages[-1].content), re.M)
            return AIMessage(content=json.dumps({"gaps": [
                self.coverage.get(k, {"point_key": k, "covered": False, "source_id": "",
                                      "quote": "", "justification": "not found"})
                for k in keys
            ]}))
        calls = self.turns.pop(0) if self.turns else []
        return AIMessage(content="" if calls else "Nothing more.", tool_calls=[
            {"name": "CreateFollowUp", "id": f"f{n}", "args": {
                "question": f"Find {key}", "gap_point_keys": [key], "reason": "missing",
                "evidence_points": [{"text": f"evidence for {key}", "required_answers": [2]}]}}
            for n, key in enumerate(calls)
        ])


COVER_WRITERS = {"t2:E1": {"point_key": "t2:E1", "covered": True, "source_id": "src_limits",
                           "quote": "there can only be one writer at a time",
                           "justification": "WAL limit"}}


def test_gap_closure_and_follow_ups_without_a_gap():
    good = score_gap_analysis(ModelRouter(GapSupervisor(COVER_WRITERS, [["t2:E2"]])), GAPS)
    assert good == {"planted_essential_gaps": 2, "closed": 2, "closure_rate": 1.0,
                    "follow_ups_without_gap": 0}
    idle = score_gap_analysis(ModelRouter(GapSupervisor({}, [])), GAPS)
    assert idle["closed"] == 0 and idle["closure_rate"] == 0.0


def test_claim_error_detection_and_false_flags():
    verdicts = {
        "mis_cited": ("unsupported", True, {"S2", "S3"}),
        "partly_supported": ("partly_supported", True, set()),
        "unsupported": ("unsupported", True, {"S1"}),
        "contradicted": ("contradicted", True, set()),
        "uncited": ("supported", True, set()),
    }
    scripted = {s["sentence"]: verdicts[s["label"]] for s in CLAIMS["sentences"] if s["label"] in verdicts}
    good = score_claims(ModelRouter(FakeClaimModel(scripted)), CLAIMS)
    planted = sum(s["failure"] is not None for s in CLAIMS["sentences"])
    assert good["planted_errors"] == planted and good["detected"] == planted
    assert good["detection_rate"] == 1.0 and good["false_flags"] == 0
    # A judge that calls everything supported still catches the rule-based errors only.
    lenient = score_claims(ModelRouter(FakeClaimModel()), CLAIMS)
    assert 0 < lenient["detected"] < planted and lenient["false_flags"] == 0


def test_compare_with_the_baseline():
    baseline = {"summary": {"repair_cycle_share": 1.0, "mean_searches_per_task": 1.0}}
    figures = compare({"repair_cycle_share": 0.5, "mean_searches_per_task": 1.5}, baseline)
    assert figures["repair_cycle_share"] == {"baseline": 1.0, "now": 0.5, "change": -0.5}
    assert figures["mean_searches_per_task"]["change"] == 0.5
    assert compare({"repair_cycle_share": None}, None)["repair_cycle_share"]["change"] is None
