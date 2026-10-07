"""The supervisor–searcher records in the UI (spec 002, US5): per-task searches and reports,
research rounds with their follow-ups, and claim verdicts that open to the exact cited text."""

import pytest

from intelligence_collector.storage.vfs import VFS
from intelligence_collector.ui.evidence import claim_views, research_rounds, task_research
from intelligence_collector.ui.html import claim_html, cited_text_html, rounds_html, task_research_html
from tests.ui.test_chat_progress import _app_with_snapshot

RUN = "run-ui"
HOSTILE = "<script>alert(1)</script>"
PAGE = "https://docs.test/limits"
SNIPPET = "https://forum.test/thread"
TODOS = [
    {"todo_id": "t1", "research_question": "SQLite maximum database size", "status": "completed"},
    {
        "todo_id": "f1",
        "research_question": f"SQLite writers {HOSTILE}",
        "status": "completed",
        "origin": "follow_up",
        "round": 2,
        "gap_point_ids": ["t1:E2"],
    },
]


@pytest.fixture(autouse=True)
def disable_local_dotenv(monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")


def _store(vfs):
    vfs.write_json(RUN, "research/t1/evaluations.json", {
        "task_id": "t1",
        "queries": ["sqlite max size", "sqlite limits"],
        "stop_reason": "all_points_covered",
        "evaluations": [
            {"url": PAGE, "decision": "accepted", "search_index": 1},
            {"url": SNIPPET, "decision": "accepted", "search_index": 1, "snippet_only": True},
            {"url": "https://spam.test", "decision": "rejected", "reason": "off_topic", "search_index": 1},
            {"url": "https://dup.test", "decision": "rejected", "reason": "duplicate", "search_index": 2},
            {"url": "https://spam2.test", "decision": "rejected", "reason": "off_topic", "search_index": 2},
        ],
    }, "result_evaluations")
    vfs.write_json(RUN, "research/t1/report.json", {
        "todo_id": "t1",
        "points": [{"point_id": "E1", "status": "answered"}, {"point_id": "E2", "status": "missing"}],
        "queries": ["sqlite max size", f"sqlite {HOSTILE}"],
        "stop_reason": "search_budget_spent",
        "report_status": "corrected",
    }, "searcher_report")
    vfs.write_json(RUN, "research/t1/evidence/result.json", {
        "task_id": "t1",
        "sufficient": False,
        "unresolved_gaps": ["E2"],
        "sources": [
            {"locator": PAGE, "provenance": {"title": "Limits"}, "excerpt": "281 TB", "raw_path": "r/0.md"},
            {"locator": SNIPPET, "provenance": {"title": "Forum"}, "excerpt": "one writer", "snippet_only": True},
        ],
    }, "evidence_bundle")
    vfs.write_json(RUN, "research/rounds/round-1.json", {
        "analysis": {
            "round": 1,
            "ended": "follow_ups_created",
            "closed_essential": 1,
            "gaps": [{"point_key": "t1:E2", "text": f"concurrent writers {HOSTILE}",
                      "essential": True, "resolution": "open", "status": "missing"}],
            "refusals": [{"reason": "cap reached for answer 2"}],
        },
        "follow_up_todos": [{"todo_id": "f1", "research_question": f"SQLite writers {HOSTILE}",
                             "gap_point_ids": ["t1:E2"]}],
    }, "gap_analysis")
    vfs.write_json(RUN, "research/rounds/round-2.json", {
        "analysis": {"round": 2, "ended": "no_essential_gaps", "gaps": []},
        "follow_up_todos": [],
    }, "gap_analysis")
    vfs.write_json(RUN, "synthesis/evidence.v1.json", {
        "S1": {"title": "Limits", "url": PAGE, "text": "The maximum database size is 281 TB."},
        "S2": {"title": "Forum", "url": SNIPPET, "text": f"Only one writer {HOSTILE} at a time."},
    }, "draft_evidence")
    vfs.write_json(RUN, "verification/claims.v1.json", {
        "draft_version": 1,
        "artifact_hash": "h",
        "supported_share": 0.5,
        "passed": False,
        "claims": [
            {"claim_id": "K1", "sentence": "SQLite holds 281 TB [S1].", "cited_source_ids": ["S1"],
             "states_fact": True, "verdict": "supported", "failures": []},
            {"claim_id": "K2", "sentence": f"Many writers {HOSTILE} [S1][S2].",
             "cited_source_ids": ["S1", "S2"], "states_fact": True, "verdict": "contradicted",
             "failures": ["source_supports_nothing:S1", "contradicted"]},
            {"claim_id": "K3", "sentence": "This section compares engines.", "cited_source_ids": [],
             "states_fact": False, "verdict": None, "failures": []},
        ],
    }, "claim_evaluation")


def test_task_research_counts_decisions_and_reads_the_report(tmp_path):
    vfs = VFS(str(tmp_path))
    _store(vfs)
    research = task_research(vfs, RUN, TODOS[0])

    assert research.accepted == 2 and research.rejected == {"off_topic": 2, "duplicate": 1}
    assert research.stop_reason == "search_budget_spent"  # the report wins over evaluations
    assert research.report_status == "corrected" and research.open_points == ["E2 missing"]
    fragment = task_research_html(research)
    assert "2 accepted · 3 rejected (duplicate 1, off topic 2)" in fragment
    assert "stopped: search budget spent · report: corrected" in fragment
    assert HOSTILE not in fragment and "&lt;script&gt;" in fragment
    assert task_research(vfs, RUN, {"todo_id": "never-ran"}) is None


def test_rounds_list_follow_ups_with_their_gaps_escaped(tmp_path):
    vfs = VFS(str(tmp_path))
    _store(vfs)
    rounds = research_rounds(vfs, RUN)

    assert [r.round for r in rounds] == [1, 2]
    assert rounds[0].follow_ups[0].gap_point_ids == ["t1:E2"]
    fragment = rounds_html(rounds)
    assert "Round 1: follow ups created · 1 essential gap(s) closed" in fragment
    assert "gaps t1:E2" in fragment and "Refused: cap reached for answer 2" in fragment
    assert "Round 2: no essential gaps" in fragment
    assert HOSTILE not in fragment


def test_claims_open_to_the_snapshot_text_and_label_snippet_only_sources(tmp_path):
    vfs = VFS(str(tmp_path))
    _store(vfs)
    snapshot = {"claim_evaluation_ref": "verification/claims.v1.json", "todos": TODOS}
    claims = claim_views(vfs, RUN, snapshot)

    # Non-factual sentences are not listed; failing claims come first.
    assert [c.claim_id for c in claims] == ["K2", "K1"]
    failing = claims[0]
    assert [s.source_id for s in failing.sources] == ["S1", "S2"]
    assert failing.sources[1].text == f"Only one writer {HOSTILE} at a time."
    assert [s.snippet_only for s in failing.sources] == [False, True]
    fragment = claim_html(failing) + "".join(cited_text_html(s) for s in failing.sources)
    assert "K2 · contradicted" in fragment and "source_supports_nothing:S1; contradicted" in fragment
    assert "snippet only" in fragment and HOSTILE not in fragment
    assert claim_views(vfs, RUN, {"todos": TODOS}) == []  # before verification


def test_the_panel_renders_research_rounds_and_claims(tmp_path, monkeypatch):
    monkeypatch.setenv("INTELLIGENCE_VFS_ROOT", str(tmp_path))
    _store(VFS(str(tmp_path)))
    app = _app_with_snapshot({
        "run_id": RUN,
        "thread_id": "thread-ui",
        "stage": "Awaiting Approval",
        "verification": {"passed": True, "artifact_hash": "h", "checks": []},
        "report": "# Report",
        "todos": TODOS,
        "claim_evaluation_ref": "verification/claims.v1.json",
        "claim_support_score": 0.5,
        "activity": [{"id": "research", "label": "Delegating research to sub-agents",
                      "agent": "Research Supervisor", "status": "success"}],
    })

    assert not app.exception
    run = "\n".join(str(m.value) for m in app.tabs[0].markdown)
    assert "Claims · 1 of 2 supported" in run
    assert "2 accepted · 3 rejected" in run
    labels = [e.label for e in app.tabs[0].expander]
    assert any(label.startswith("✗ K2") for label in labels)
    assert any(label.startswith("Follow-up (round 2)") and "t1:E2" in label for label in labels)
    assert "The maximum database size is 281 TB." in [t.value for t in app.tabs[0].text]
    assert "snippet only" in "\n".join(c.value for c in app.tabs[0].caption)
    activity = "\n".join(str(m.value) for m in app.tabs[2].markdown)
    assert "Research rounds" in activity and "Follow-up: SQLite writers" in activity
    assert HOSTILE not in run + activity
