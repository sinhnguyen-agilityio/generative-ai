"""Scripted models, search and store, so the full app runs offline without API keys.

Used by the tests and by the run skill's offline Streamlit entry point
(.claude/skills/run-deep-research/offline_app.py). Keep it free of pytest imports.
"""

from itertools import count
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver

from deep_research.config import ModelSpec, Settings
from deep_research.orchestrator import build_graph
from deep_research.runner import DeepResearch
from deep_research.state import ClarifyWithUser, Evaluation, ResearchBrief, ResearchPlan, SafetyVerdict


class FakeTier:
    """``structured`` answers by schema name; ``tools`` and ``plain`` answer from the messages."""

    def __init__(self, structured: dict, tools=None, plain=None):
        self._structured = structured
        self._tools = tools
        self._plain = plain

    def structured(self, schema):
        def answer(_):
            value = self._structured[schema.__name__]
            if isinstance(value, Exception):
                raise value
            return value

        return RunnableLambda(answer)

    def tools(self, _tools):
        return RunnableLambda(self._tools)

    def plain(self):
        return RunnableLambda(lambda prompt: AIMessage(self._plain(prompt)))


def call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def supervisor(messages):
    if any(isinstance(m, ToolMessage) for m in messages):
        return AIMessage("", tool_calls=[call("ResearchComplete", {}, "done")])
    return AIMessage("", tool_calls=[
        call("Think", {"reflection": "two topics"}, "t1"),
        call("ConductResearch", {"research_topic": "topic A"}, "r1"),
        call("ConductResearch", {"research_topic": "topic B"}, "r2"),
    ])


def researcher(messages):
    if isinstance(messages[-1], ToolMessage):
        return AIMessage("enough")
    return AIMessage("", tool_calls=[call("WebSearch", {"query": "q"}, "s1")])


class FakeSearch:
    def search(self, query):
        return [{"url": "https://example.com", "title": "Example", "content": "fact", "raw_content": "fact"}]


class FakeStore:
    def __init__(self):
        self.added = []

    def add(self, results, *, run_id, topic):
        self.added.append((run_id, topic))


def make_app(output_dir: Path, clarify=False, evaluation=None, safe=True):
    """A ``DeepResearch`` app on scripted models; returns ``(app, store)``. Reports are
    "# Report v1", "# Report v2", ... in the order they are written."""
    settings = Settings(reasoning=ModelSpec("fake"), standard=ModelSpec("fake"), output_dir=str(output_dir))
    reasoning = FakeTier(
        {
            "ClarifyWithUser": ClarifyWithUser(need_clarification=clarify, question="Which year?", verification="Starting."),
            "ResearchBrief": ResearchBrief(brief="Research X."),
            "ResearchPlan": ResearchPlan(topics=["topic A", "topic B"]),
            "Evaluation": evaluation or Evaluation(faithfulness=0.9, relevance=0.8, comment="ok"),
        },
        tools=supervisor,
    )
    reports = count(1)
    verdict = SafetyVerdict(allowed=safe, reason="" if safe else "Asks for weapon instructions.")
    # The writer gets a prompt string; compression gets the researcher's messages.
    standard = FakeTier(
        {"SafetyVerdict": verdict},
        tools=researcher,
        plain=lambda p: f"# Report v{next(reports)}" if isinstance(p, str) else "findings [1]",
    )
    store = FakeStore()
    graph = build_graph(settings, reasoning, standard, FakeSearch(), store, checkpointer=MemorySaver())
    return DeepResearch(settings, graph=graph), store
