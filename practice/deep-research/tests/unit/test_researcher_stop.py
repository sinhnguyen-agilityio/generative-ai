"""Searcher stop conditions and query steering (spec 002, FR-004, FR-006, FR-007)."""

import json

from langchain_core.messages import AIMessage, ToolMessage

from intelligence_collector.agents.researcher import ResearchAgent
from intelligence_collector.contracts import EvidencePoint, ResearchTodo
from intelligence_collector.models.router import ModelRouter
from intelligence_collector.tools.mcp import ControlledMCP

TODO = ResearchTodo(
    todo_id="t1",
    research_question="What are SQLite's size and writer limits?",
    evidence_points=[
        EvidencePoint(point_id="E1", text="maximum database size"),
        EvidencePoint(point_id="E2", text="concurrent writers"),
    ],
)


class Searcher:
    """Scripted searcher + result evaluator. `queries` are issued one per turn (then a final
    note); `covers` gives, per search, the evidence points every accepted row covers, or
    "reject" to reject every row, or "fail" to fail the evaluation."""

    model_name = "fake-searcher"

    def __init__(self, queries, covers=None):
        self.queries = list(queries)
        self.covers = list(covers or [])
        self.evaluations = 0
        self.prompts = []

    def bind_tools(self, tools, tool_choice=None):
        return self

    def invoke(self, messages, config=None):
        if str(messages[0].content).startswith("You evaluate web search results"):
            plan = self.covers[self.evaluations] if self.evaluations < len(self.covers) else []
            self.evaluations += 1
            if plan == "fail":
                raise RuntimeError("evaluator unavailable")
            rows = str(messages[-1].content).count("\nURL: ")
            if plan == "reject":
                return AIMessage(content=json.dumps({"rows": [
                    {"index": i, "decision": "rejected", "reason": "off_topic"}
                    for i in range(1, rows + 1)
                ]}))
            return AIMessage(content=json.dumps({"rows": [
                {"index": i, "decision": "accepted", "covers": plan} for i in range(1, rows + 1)
            ]}))
        self.prompts.append(messages)
        done = sum(isinstance(m, ToolMessage) for m in messages)
        if done < len(self.queries):
            return AIMessage(content="", tool_calls=[
                {"name": "tavily_search", "args": {"query": self.queries[done]}, "id": f"c{done}"}
            ])
        return AIMessage(content="Done.")


def pages(n, prefix="p"):
    return [[{"url": f"https://{prefix}.test/{i}", "content": f"page {i}"}] for i in range(n)]


def run(model, mcp, *, max_sources=5):
    agent = ResearchAgent(ModelRouter(model), lambda: mcp, max_sources=max_sources)
    return agent.search(TODO.research_question, todo=TODO, task_id=TODO.todo_id)


def test_stops_when_every_evidence_point_is_covered():
    outcome = run(Searcher(["q one", "q two", "q three"], covers=[["E1"], ["E2"]]), ControlledMCP(pages(5)))
    assert outcome.stop_reason == "all_points_covered"
    assert outcome.search_calls == 2 and outcome.covered_points == ["E1", "E2"]


def test_stops_after_two_searches_without_an_accepted_result():
    model = Searcher(["a", "b", "c", "d"], covers=[["E1"], "reject", "reject"])
    outcome = run(model, ControlledMCP(pages(5)))
    assert outcome.stop_reason == "no_progress" and outcome.search_calls == 3
    assert len(outcome.sources) == 1


def test_failed_result_evaluations_count_as_failed_searches():
    # The router retries a failed structured call once, so every attempt must fail.
    outcome = run(Searcher(["a", "b", "c"], covers=["fail"] * 6), ControlledMCP(pages(5)))
    assert outcome.stop_reason == "search_failures" and outcome.search_calls == 2
    assert outcome.sources == [] and outcome.failure_kind == "technical"


def test_search_budget_and_page_budget():
    spent = run(Searcher([f"q{n}" for n in range(8)], covers=[[]] * 8), ControlledMCP(pages(8)), max_sources=10)
    assert spent.stop_reason == "search_budget_spent" and spent.search_calls == 5
    full = run(Searcher([f"q{n}" for n in range(8)], covers=[[]] * 8), ControlledMCP(pages(8)), max_sources=2)
    assert full.stop_reason == "page_budget_full" and len(full.sources) == 2


def test_model_finishing_on_its_own_and_content_failure_kind():
    outcome = run(Searcher(["only query"], covers=["reject"]), ControlledMCP(pages(2)))
    assert outcome.stop_reason == "model_finished"
    assert outcome.failure_kind == "content"


def test_repeated_query_is_refused_without_spending_the_budget():
    model = Searcher(["SQLite size limit", "limit size sqlite", "sqlite writers"], covers=[["E1"], ["E2"]])
    outcome = run(model, ControlledMCP(pages(5)))
    assert outcome.queries == ["SQLite size limit", "sqlite writers"]
    assert outcome.search_calls == 2
    refused = [m for m in model.prompts[-1] if isinstance(m, ToolMessage)]
    assert any("repeats an earlier query" in str(m.content) for m in refused)


def test_tool_reply_lists_rejections_and_uncovered_points():
    model = Searcher(["a", "b"], covers=[["E1"], "reject"])
    run(model, ControlledMCP(pages(3)))
    replies = [str(m.content) for m in model.prompts[-1] if isinstance(m, ToolMessage)]
    assert "ACCEPTED" in replies[0] and "- concurrent writers" in replies[0]
    assert "REJECTED (off_topic)" in replies[1]
