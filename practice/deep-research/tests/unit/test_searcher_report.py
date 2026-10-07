"""Searcher gap self-lookup and short report (spec 002, FR-013 to FR-014)."""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from intelligence_collector.agents.researcher import ResearchAgent, ResearchOutcome
from intelligence_collector.agents.search_task import SearchTask
from intelligence_collector.agents.searcher_report import (
    StoredSource,
    lookup_points,
    validate_report,
    write_report,
)
from intelligence_collector.contracts import (
    EvidencePoint,
    IndexedChunk,
    ResearchTodo,
    SearcherReportDraft,
)
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.storage.passage_lookup import PassageLookup
from intelligence_collector.storage.vector_index import LocalVectorIndex
from intelligence_collector.storage.vfs import VFS
from intelligence_collector.tools.mcp import ControlledMCP
from intelligence_collector.tools.url_tracking import UrlRegistry
from tests.fixtures.fake_models import FakeResearchModel

TODO = ResearchTodo(
    todo_id="t1",
    research_question="What are SQLite's limits?",
    evidence_points=[
        EvidencePoint(point_id="E1", text="maximum database size", required_answers=["size"]),
        EvidencePoint(point_id="E2", text="concurrent writers", required_answers=["writers"]),
    ],
)
PAGE = (
    "The maximum database size is 281 TB. In WAL mode there can only be one writer at a time, "
    "while many readers proceed concurrently."
)
SOURCES = {"src_a": StoredSource("src_a", "https://sqlite.test/limits", "Limits", PAGE, [PAGE])}


def outcome(**extra):
    base = {"question": TODO.research_question, "search_budget": 5, "mcp_server": "c", "mcp_tool": "s"}
    return ResearchOutcome(**{**base, **extra})


def validate(draft, **extra):
    common = dict(
        todo=TODO,
        sources=SOURCES,
        stop_reason="all_points_covered",
        failure_kind=None,
        queries=["q"],
        accepted=1,
        rejected=0,
        lookup_used=set(),
    )
    return validate_report(SearcherReportDraft.model_validate(draft), **{**common, **extra})


def test_verified_quotes_pass_and_the_report_is_complete():
    report = validate(
        {
            "points": [
                {"point_id": "E1", "status": "answered",
                 "support": [{"source_id": "src_a", "quote": "maximum database size is 281 TB"}]},
                {"point_id": "E2", "status": "partial",
                 "support": [{"source_id": "src_a", "quote": "there can only be one writer at a time"}]},
            ],
            "findings": [{"text": "SQLite allows one writer.", "source_ids": ["src_a"]}],
        }
    )
    assert [p.status.value for p in report.points] == ["answered", "partial"]
    assert report.report_status.value == "complete" and report.corrections == []


def test_invalid_parts_become_missing_with_corrections():
    report = validate(
        {
            "points": [
                {"point_id": "E1", "status": "answered",
                 "support": [{"source_id": "src_a", "quote": "databases may grow without any limit"}]},
                {"point_id": "E9", "status": "answered", "support": []},
            ],
            "findings": [
                {"text": "From a rejected page.", "source_ids": ["src_rejected"]},
                {"text": "Kept.", "source_ids": ["src_a"]},
            ],
        }
    )
    assert [p.status.value for p in report.points] == ["missing", "missing"]
    assert [f.text for f in report.findings] == ["Kept."]
    assert report.report_status.value == "corrected"
    text = " | ".join(report.corrections)
    assert "quote not found in src_a" in text and "E9: not an evidence point" in text
    assert "E2: missing from the report" in text and "unknown or rejected source" in text


def test_long_quotes_are_cut_and_findings_trimmed_to_the_word_limit():
    text = f"{PAGE} {PAGE}"
    long_quote = " ".join(text.split()[:35])
    findings = [{"text": "word " * 150, "source_ids": ["src_a"]} for _ in range(3)]
    report = validate(
        {
            "points": [{"point_id": "E1", "status": "answered",
                        "support": [{"source_id": "src_a", "quote": long_quote}]}],
            "findings": findings,
        },
        sources={"src_a": StoredSource("src_a", "https://sqlite.test", "Limits", text, [text])},
    )
    quote = report.points[0].support[0].quote
    assert len(quote.split()) == 30 and quote in text
    assert len(report.findings) == 2  # 2 × 150 + quote ≤ 400 words
    assert any("cut to 30 words" in c for c in report.corrections)


def test_unreadable_report_marks_every_point_missing():
    class Broken:
        model_name = "broken"

        def invoke(self, messages, config=None):
            raise RuntimeError("bad gateway; api_key=sk-abc123")

    report = write_report(
        TODO, TODO.research_question, SOURCES, ModelRouter(Broken()),
        outcome=outcome(stop_reason="model_finished"), lookup_used=set(),
    )
    assert report.report_status.value == "unreadable"
    assert all(p.status.value == "missing" for p in report.points)
    assert "sk-abc123" not in " ".join(report.corrections)


def test_lookup_without_phrasings_uses_the_combined_query_only(tmp_path):
    index = LocalVectorIndex(tmp_path / "index.json")
    index.index([IndexedChunk(chunk_id="r:src_a:0", source_id="src_a", artifact_id="a", text=PAGE,
                              metadata={"run_id": "r", "locator": "https://sqlite.test/limits"})])
    found = lookup_points(
        TODO.evidence_points[:1], todo=TODO, question=TODO.research_question,
        lookup=PassageLookup(index), run_id="r", allowed_source_ids={"src_a"},
        phrasings={}, phrasing_failed=True, requested_by="searcher:t1",
    )
    record = found.records[0]
    assert record.fallback == "combined_query_only"
    assert record.queries == ["maximum database size — What are SQLite's limits?"]
    assert found.points == {"E1"} and found.passages["src_a"] == [PAGE]


class StoredAnswerModel(FakeResearchModel):
    """A searcher that, when stored passages answer its points, finishes without searching."""

    def invoke(self, messages, config=None):
        opening = next((m for m in messages if isinstance(m, HumanMessage)), None)
        if (
            str(messages[0].content).startswith("You are a research agent")
            and opening is not None
            and "Passages the run already stored" in str(opening.content)
        ):
            return AIMessage(content="The stored passages answer every point.")
        return super().invoke(messages, config)


def test_a_point_answered_by_a_finished_tasks_page_needs_no_web_search(tmp_path):
    vfs, index = VFS(tmp_path / "vfs"), LocalVectorIndex(tmp_path / "index.json")
    # A finished task t0 of the same plan stored this page (deep in it: the writer limit).
    index.index([IndexedChunk(chunk_id="run:src_b:0", source_id="src_b", artifact_id="art-b",
                              text="Write-ahead log. " + PAGE,
                              metadata={"run_id": "run", "todo_id": "t0",
                                        "locator": "https://sqlite.test/wal"})])
    registry = UrlRegistry()
    registry.register("https://sqlite.test/wal", {"source_id": "src_b", "task_id": "t0",
                                                  "artifact_id": "art-b", "content_hash": "h",
                                                  "locator": "https://sqlite.test/wal"})
    registry.save(vfs, "run")
    mcp = ControlledMCP([{"url": "https://web.test/x", "content": "unused"}])
    researcher = ResearchAgent(ModelRouter(StoredAnswerModel()), lambda: mcp)
    task = SearchTask(vfs, index, researcher, "run", TODO, TODO.research_question, {"t0", "t1"},
                      lookup=PassageLookup(index))

    todo, refs, _ = task.run()

    assert mcp.calls == 0  # every point had stored passages: the first search was not forced
    report = json.loads(vfs.read("run", "research/t1/report.json"))
    assert [p["status"] for p in report["points"]] == ["answered", "answered"]
    assert all(p["lookup_used"] for p in report["points"])
    assert report["points"][0]["support"][0]["source_id"] == "src_b"
    lookups = json.loads(vfs.read("run", "research/t1/lookups.json"))
    assert {r["requested_by"] for r in lookups} == {"searcher:t1"}
    bundle = json.loads(vfs.read("run", "research/t1/evidence/result.json"))
    assert todo.status.value == "completed" and bundle["sufficient"] is True
    (source,) = bundle["sources"]
    assert source["locator"] == "https://sqlite.test/wal" and source["reused_from_task"] == "t0"
    assert "one writer at a time" in source["excerpt"]


def test_superseded_tasks_pages_are_never_looked_up(tmp_path):
    vfs, index = VFS(tmp_path / "vfs"), LocalVectorIndex(tmp_path / "index.json")
    index.index([IndexedChunk(chunk_id="run:src_old:0", source_id="src_old", artifact_id="x",
                              text=PAGE, metadata={"run_id": "run", "todo_id": "old",
                                                   "locator": "https://old.test"})])
    registry = UrlRegistry()
    registry.register("https://old.test", {"source_id": "src_old", "task_id": "old"})
    registry.save(vfs, "run")
    task = SearchTask(vfs, index, ResearchAgent(ModelRouter(FakeResearchModel()), lambda: None),
                      "run", TODO, TODO.research_question, {"t1"}, lookup=PassageLookup(index))
    task.lookup_before_search()
    assert task.found.points == set() and task.found.records == []


@pytest.mark.parametrize("words", [100, 400])
def test_word_limit_is_configurable(words):
    findings = [{"text": "word " * 60, "source_ids": ["src_a"]} for _ in range(5)]
    report = validate({"points": [], "findings": findings}, max_words=words)
    assert sum(len(f.text.split()) for f in report.findings) <= words
