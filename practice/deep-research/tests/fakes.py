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
from deep_research.state import (
    Answered, ClarifyWithUser, ResearchBrief, ResearchPlan, SafetyVerdict, TopicReport,
)


class FakeTier:
    """``structured`` answers by schema name (a callable answers from the prompt); ``tools`` and
    ``plain`` answer from the messages."""

    def __init__(self, structured: dict, tools=None, plain=None):
        self._structured = structured
        self._tools = tools
        self._plain = plain

    def structured(self, schema):
        def answer(prompt):
            value = self._structured[schema.__name__]
            if isinstance(value, Exception):
                raise value
            return value(prompt) if callable(value) else value

        return RunnableLambda(answer)

    def tools(self, _tools):
        return RunnableLambda(self._tools)

    def plain(self):
        return RunnableLambda(lambda prompt: AIMessage(self._plain(prompt)))


def call(name: str, args: dict, call_id: str) -> dict:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def supervisor(messages):
    """Called only after the plan is researched: the reports are enough."""
    return AIMessage("", tool_calls=[call("ResearchComplete", {}, "done")])


def researcher(messages):
    if isinstance(messages[-1], ToolMessage):
        return AIMessage("enough")
    return AIMessage("", tool_calls=[call("WebSearch", {"query": "q"}, "s1")])


# A page long enough to go through retrieval, and a snippet long enough to be kept.
PAGE = "\n".join(f"Fact {i}: Gemini Flash input costs ${i}.{i:02d} per million tokens in region {i}." for i in range(150))
SNIPPET = "Gemini Flash input costs $0.30 per million tokens and output costs $2.50 per million tokens. " * 3


def result(url="https://example.com/a", raw_content=PAGE, content=SNIPPET):
    return {"url": url, "title": f"Page {url[-1]}", "content": content, "raw_content": raw_content, "score": 0.9}


class FakeSearch:
    def __init__(self, results=None, error=None):
        self.results = [result()] if results is None else results
        self.error = error

    def search(self, query):
        if self.error:
            raise self.error
        return self.results


class FakeStore:
    """Keeps added page texts; ``query`` returns the first characters of each requested page."""

    def __init__(self, fail=False):
        self.added, self.queries, self.pages, self.fail = [], [], {}, fail

    def add(self, pages, *, run_id, topic):
        if self.fail:
            raise RuntimeError("chroma down")
        self.added.append((run_id, topic))
        self.pages.update({(run_id, p["url"]): p["text"] for p in pages})

    def query(self, text, *, run_id, urls, k):
        self.queries.append(text)
        return [{"url": url, "text": self.pages[(run_id, url)][:1000]} for url in urls][:k]


TOPIC_REPORT = TopicReport(
    findings="Gemini Flash input costs $0.30 per million tokens [1].",
    answered=[Answered(question="What is the input price?", citations=[1])],
    missing_parts=[],
    coverage="full",
)


def fake_scorer(scores):
    """Stands in for the RAGAS scorer: returns ``scores``, or raises it when it is an exception."""
    def score(brief, report, contexts):
        if isinstance(scores, Exception):
            raise scores
        return scores

    return score


def make_app(output_dir: Path, clarify=False, evaluation=None, safe=True, topics=("topic A", "topic B")):
    """A ``DeepResearch`` app on scripted models; returns ``(app, store)``. Reports are
    "# Report v1", "# Report v2", ... in the order they are written."""
    settings = Settings(reasoning=ModelSpec("fake"), standard=ModelSpec("fake"), output_dir=str(output_dir))
    reasoning = FakeTier(
        {
            "ClarifyWithUser": ClarifyWithUser(need_clarification=clarify, question="Which year?", verification="Starting."),
            "ResearchBrief": ResearchBrief(brief="Research X."),
            "ResearchPlan": ResearchPlan(topics=list(topics)),
        },
        tools=supervisor,
    )
    reports = count(1)
    verdict = SafetyVerdict(allowed=safe, reason="" if safe else "Asks for weapon instructions.")
    standard = FakeTier(
        {"SafetyVerdict": verdict, "TopicReport": TOPIC_REPORT},
        tools=researcher,
        plain=lambda _: f"# Report v{next(reports)}",
    )
    store = FakeStore()
    scorer = fake_scorer(evaluation or {"faithfulness": 0.9, "answer_relevancy": 0.8})
    graph = build_graph(settings, reasoning, standard, FakeSearch(), store, checkpointer=MemorySaver(), scorer=scorer)
    return DeepResearch(settings, graph=graph), store
