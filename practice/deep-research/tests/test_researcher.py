"""The researcher: search → index → retrieve inside WebSearch, and the topic report checked in code."""

import asyncio

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from deep_research.agents.researcher import ResearcherAgent, render_note
from deep_research.config import ModelSpec, Settings
from deep_research.search import clean, normalize_url, prepare_pages
from deep_research.state import Answered, MissingPart, TopicReport
from tests.fakes import PAGE, SNIPPET, TOPIC_REPORT, FakeSearch, FakeStore, FakeTier, call, researcher, result


def run(search=None, store=None, report=TOPIC_REPORT, tools=researcher):
    """Run one researcher inside a thread config, as the supervisor does; returns
    ``(topic report, the WebSearch results the model saw, store)``."""
    seen = []

    def reporter(messages):
        seen.extend(m.text for m in messages if isinstance(m, ToolMessage) and m.name == "WebSearch")
        if isinstance(report, Exception):
            raise report
        return report

    store = store or FakeStore()
    settings = Settings(reasoning=ModelSpec("fake"), standard=ModelSpec("fake"))
    agent = ResearcherAgent(FakeTier({"TopicReport": reporter}, tools=tools), search or FakeSearch(), store, settings)
    note = asyncio.run(RunnableLambda(agent.research).ainvoke("topic A", {"configurable": {"thread_id": "t"}}))
    return render_note(note), seen, store


def test_report_has_the_four_criteria_and_code_built_citations():
    notes, seen, store = run()
    assert notes.startswith("## Status: complete\n## Coverage: full")
    assert "- What is the input price? [1]" in notes
    assert "## Missing parts\n- None" in notes
    assert notes.endswith("1 sources, 1 domains, 0 snippet only\n[1] Page a: https://example.com/a")
    assert store.queries == ["q"]  # the long page went through retrieval
    assert seen[0].startswith("[1] Page a — https://example.com/a (full text)\nFact 0:")


def test_cited_sources_carry_the_passages_the_model_saw():
    settings = Settings(reasoning=ModelSpec("fake"), standard=ModelSpec("fake"))
    agent = ResearcherAgent(FakeTier({"TopicReport": TOPIC_REPORT}, tools=researcher), FakeSearch(), FakeStore(), settings)
    note = asyncio.run(RunnableLambda(agent.research).ainvoke("topic A", {"configurable": {"thread_id": "t"}}))
    [source] = note["sources"]
    assert source["url"] == "https://example.com/a"
    assert source["passages"] == [PAGE[:1000]]  # the retrieved passage shown to the model


def test_a_page_seen_again_keeps_its_source_number():
    def two_searches(messages):
        done = sum(isinstance(m, ToolMessage) for m in messages)
        if done < 2:
            return AIMessage("", tool_calls=[call("WebSearch", {"query": f"q{done}"}, f"s{done}")])
        return AIMessage("enough")

    search = FakeSearch([result("https://Example.com/a/?utm_source=x"), result("https://example.com/b")])
    _, seen, _ = run(search=search, tools=two_searches)
    assert len(seen) == 2
    for text in seen:
        assert "[1] Page x — https://example.com/a (full text)" in text
        assert "[2] Page b — https://example.com/b (full text)" in text


def test_missing_raw_content_falls_back_to_the_snippet_and_short_results_are_dropped():
    search = FakeSearch([result("https://example.com/a", raw_content=""), result("https://example.com/b", "", "tiny")])
    notes, seen, store = run(search=search)
    assert seen == [f"[1] Page a — https://example.com/a (snippet only)\n{SNIPPET.strip()}"]
    assert store.queries == []  # short results are shown whole
    assert "1 sources, 1 domains, 1 snippet only" in notes


def test_unknown_citations_are_dropped_and_leave_the_report_failed():
    report = TopicReport(
        findings="A claim [9].", answered=[Answered(question="Price?", citations=[9])], missing_parts=[], coverage="full"
    )
    notes, _, _ = run(report=report)
    assert notes.startswith("## Status: failed\n## Coverage: minimal")
    assert "[9]" not in notes and "A claim" not in notes
    assert "- Price? (essential: no valid source)" in notes


def test_sentences_citing_only_unknown_sources_are_dropped():
    report = TOPIC_REPORT.model_copy(update={"findings": "- Valid [1]. Made up [9].\n- Mixed [1][9]. Plain sentence."})
    notes, _, _ = run(report=report)
    assert "## Findings\n- Valid [1].\n- Mixed [1]. Plain sentence.\n## Citations" in notes


def test_an_essential_missing_part_caps_coverage_and_status():
    report = TOPIC_REPORT.model_copy(
        update={"missing_parts": [MissingPart(question="Free tier?", essential=True, reason="not found")]}
    )
    notes, _, _ = run(report=report)
    assert notes.startswith("## Status: partial\n## Coverage: partial")
    assert "- Free tier? (essential: not found)" in notes


def test_a_missing_part_listed_twice_is_kept_once():
    parts = [
        MissingPart(question="LangSmith pricing?", essential=False, reason="not found"),
        MissingPart(question="  langsmith   PRICING? ", essential=True, reason="not found"),
    ]
    notes, _, _ = run(report=TOPIC_REPORT.model_copy(update={"missing_parts": parts}))
    assert notes.count("PRICING?") + notes.count("pricing?") == 1
    assert "(essential: not found)" in notes


def test_a_store_failure_shows_the_search_snippets():
    notes, seen, _ = run(store=FakeStore(fail=True))
    assert seen == [f"[1] Page a — https://example.com/a (full text)\n{SNIPPET.strip()}"]
    assert notes.startswith("## Status: complete")


def test_a_search_failure_gives_a_failed_report():
    notes, seen, _ = run(search=FakeSearch(error=RuntimeError("Tavily down")))
    assert seen == ["Search failed: Tavily down"]
    assert notes.startswith("## Status: failed")


def test_a_failed_report_call_gives_a_failed_report():
    notes, _, _ = run(report=RuntimeError("bad json"))
    assert notes.startswith("## Status: failed")
    assert "Topic report failed: bad json" in notes


def test_normalize_url():
    assert normalize_url("HTTPS://Example.com/a/?utm_source=x&id=1#top") == "https://example.com/a?id=1"


def test_results_for_the_same_page_become_one_page():
    pages = prepare_pages([result("https://example.com/a"), result("https://example.com/a/?utm_source=x")])
    assert [p["url"] for p in pages] == ["https://example.com/a"]


def test_clean_drops_page_noise_and_keeps_content():
    page = "\n".join([
        "Home", "Sign in", "[Docs](https://x.dev/docs)", "We use cookies to improve your experience.",
        "## Pricing", "Input price: $0.30 per 1M tokens", "Input price: $0.30 per 1M tokens",
        "Gemini Flash is the fastest model in the family.",
    ])
    assert clean(page) == "## Pricing\nInput price: $0.30 per 1M tokens\nGemini Flash is the fastest model in the family."


def test_searches_in_one_turn_run_concurrently():
    class ParallelSearch(FakeSearch):
        def __init__(self):
            super().__init__()
            self.active = self.peak = 0

        async def search(self, query):
            self.active += 1
            self.peak = max(self.peak, self.active)
            await asyncio.sleep(0)
            self.active -= 1
            return self.results

    def tools(messages):
        if isinstance(messages[-1], ToolMessage):
            return AIMessage("enough")
        return AIMessage("", tool_calls=[call("WebSearch", {"query": q}, q) for q in ("q1", "q2")])

    search = ParallelSearch()
    notes, seen, _ = run(search=search, tools=tools)
    assert search.peak == 2
    assert len(seen) == 2
    assert "## Status: complete" in notes


def test_researcher_decides_when_to_stop_beyond_prompt_budget():
    def tools(messages):
        searches = sum(isinstance(m, ToolMessage) and m.name == "WebSearch" for m in messages)
        if searches == 6:
            return AIMessage("enough")
        return AIMessage("", tool_calls=[call("WebSearch", {"query": f"q{searches}"}, f"s{searches}")])

    _, seen, _ = run(tools=tools)
    assert len(seen) == 6
