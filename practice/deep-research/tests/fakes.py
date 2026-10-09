"""Scripted models, search and store, so the full app runs offline without API keys.

Used by the tests and by the run skill's offline Streamlit entry point
(.claude/skills/run-deep-research/offline_app.py). Keep it free of pytest imports.
"""

from itertools import count
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver

from deep_research.model_spec import ModelSpec
from deep_research.orchestrator import build_graph
from deep_research.runner import DeepResearch
from deep_research.schemas.answered import Answered
from deep_research.schemas.clarify_with_user import ClarifyWithUser
from deep_research.schemas.follow_up import FollowUp
from deep_research.schemas.report_check import ReportCheck
from deep_research.schemas.research_brief import ResearchBrief
from deep_research.schemas.research_plan import ResearchPlan
from deep_research.schemas.safety_verdict import SafetyVerdict
from deep_research.schemas.topic_report import TopicReport
from deep_research.settings import Settings


class FakeTier:
    """``structured`` answers by schema name (a callable answers from the prompt); ``tools`` and
    ``plain`` answer from the messages."""

    def __init__(self, structured: dict, tools=None, plain=None):
        """Store scripted structured, tool-calling, and plain responses.

        Args:
            structured: Schema-name mapping to values, exceptions, or response callables.
            tools: Optional callable generating tool-calling messages.
            plain: Optional callable generating plain response text.
        """
        self._structured = structured
        self._tools = tools
        self._plain = plain

    def structured(self, schema):
        """Build a runnable returning the configured result for the requested schema.

        Args:
            schema: Output schema selecting a scripted response by class name.

        Returns:
            Runnable evaluating the response callable or raising the configured exception.
        """
        def answer(prompt):
            """Resolve the scripted schema response for this prompt.

            Args:
                prompt: Input passed to an optional response callable.

            Returns:
                Configured value or callable result.
            """
            value = self._structured[schema.__name__]
            if isinstance(value, Exception):
                raise value
            return value(prompt) if callable(value) else value

        return RunnableLambda(answer)

    def tools(self, _tools):
        """Return the scripted tool-calling model without binding real tool schemas.

        Args:
            _tools: Production tool definitions accepted for API compatibility.

        Returns:
            Runnable using the configured tool-response callable.
        """
        return RunnableLambda(self._tools)

    def plain(self):
        """Wrap scripted plain text as an assistant message.

        Returns:
            Runnable returning an AIMessage from the configured text callable.
        """
        return RunnableLambda(lambda prompt: AIMessage(self._plain(prompt)))


def call(name: str, args: dict, call_id: str) -> dict:
    """Build a LangChain tool-call record for a scripted model.

    Args:
        name: Tool name.
        args: Tool argument dictionary.
        call_id: Identifier linking the call to its reply.

    Returns:
        Tool-call dictionary with name, args, id, and type.
    """
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def supervisor(messages):
    """Delegate the initial plan, finish after results, and research one follow-up topic."""
    if isinstance(messages[-1], HumanMessage) and "<Initial plan>" in messages[-1].text:
        plan = messages[-1].text.split("<Initial plan>")[1].split("</Initial plan>")[0]
        topics = [line[2:] for line in plan.strip().splitlines() if line.startswith("- ")]
        return AIMessage("", tool_calls=[call("ConductResearch", {"research_topic": t}, f"plan-{i}") for i, t in enumerate(topics, 1)])
    if isinstance(messages[-1], HumanMessage):
        return AIMessage("", tool_calls=[call("ConductResearch", {"research_topic": "follow-up"}, "more")])
    return AIMessage("", tool_calls=[call("ResearchComplete", {}, "done")])


def researcher(messages):
    """Request one web search, then finish after receiving its tool reply.

    Args:
        messages: Researcher conversation used to determine whether search has completed.

    Returns:
        Scripted WebSearch call or a final assistant message.
    """
    if isinstance(messages[-1], ToolMessage):
        return AIMessage("enough")
    return AIMessage("", tool_calls=[call("WebSearch", {"query": "q"}, "s1")])


# A page long enough to go through retrieval, and a snippet long enough to be kept.
PAGE = "\n".join(f"Fact {i}: Gemini Flash input costs ${i}.{i:02d} per million tokens in region {i}." for i in range(150))
SNIPPET = "Gemini Flash input costs $0.30 per million tokens and output costs $2.50 per million tokens. " * 3


def result(url="https://example.com/a", raw_content=PAGE, content=SNIPPET):
    """Build a normalized search result with configurable page text and snippet.

    Args:
        url: Result page URL.
        raw_content: Full page content supplied by the fake search.
        content: Fallback search snippet.

    Returns:
        Search result dictionary matching the TavilySearch contract.
    """
    return {"url": url, "title": f"Page {url[-1]}", "content": content, "raw_content": raw_content, "score": 0.9}


class FakeSearch:
    def __init__(self, results=None, error=None):
        """Configure deterministic search results or a simulated failure.

        Args:
            results: Result records; None selects the default example page.
            error: Optional exception raised by search.
        """
        self.results = [result()] if results is None else results
        self.error = error

    async def search(self, query):
        """Return configured results or raise the simulated search error.

        Args:
            query: Search query accepted for backend compatibility.

        Returns:
            Configured result list.
        """
        if self.error:
            raise self.error
        return self.results


class FakeStore:
    """Keeps added page texts; ``query`` returns the first characters of each requested page."""

    def __init__(self, fail=False):
        """Initialize recorded indexing/query calls and optional indexing failure.

        Args:
            fail: Whether add should simulate a store failure.
        """
        self.added, self.queries, self.pages, self.fail = [], [], {}, fail

    def add(self, pages, *, run_id, topic):
        """Record indexing metadata and retain page text by run and URL.

        Args:
            pages: Prepared result pages.
            run_id: Thread identifier isolating stored pages.
            topic: Delegated topic recorded with the indexing call.
        """
        if self.fail:
            raise RuntimeError("chroma down")
        self.added.append((run_id, topic))
        self.pages.update({(run_id, p["url"]): p["text"] for p in pages})

    def query(self, text, *, run_id, urls, k):
        """Record retrieval and return the first 1,000 characters of each selected page.

        Args:
            text: Query recorded for assertions.
            run_id: Thread identifying the stored pages.
            urls: Requested page URLs in return order.
            k: Maximum passages to return.

        Returns:
            URL/text passages from the fake store.
        """
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
        """Return fixed evaluation scores or raise the configured exception.

        Args:
            brief: Brief accepted for scorer compatibility.
            report: Report accepted for scorer compatibility.
            contexts: Evidence accepted for scorer compatibility.

        Returns:
            Configured metric dictionary.
        """
        if isinstance(scores, Exception):
            raise scores
        return scores

    return score


PASSED = ReportCheck(sentences=[], criteria=[])


def triage(prompt):
    """A follow-up that mentions research needs research; anything else is a rewrite."""
    message = prompt.split("<Message>")[1].split("</Message>")[0]
    return FollowUp(kind="research" if "research" in message.lower() else "revise")


def make_app(
    output_dir: Path, clarify=False, evaluation=None, safe=True, topics=("topic A", "topic B"), check=PASSED, writer=None, supervisor_model=None
):
    """A ``DeepResearch`` app on scripted models; returns ``(app, store)``. Reports are
    "# Report v1", "# Report v2", ... in the order they are written, unless ``writer(prompt)``
    writes them. ``check`` is the verifier's verdict (a callable answers from the prompt)."""
    settings = Settings(reasoning=ModelSpec("fake"), standard=ModelSpec("fake"), output_dir=str(output_dir))
    reasoning = FakeTier(
        {
            "ClarifyWithUser": ClarifyWithUser(need_clarification=clarify, question="Which year?", verification="Starting."),
            "ResearchBrief": ResearchBrief(brief="Research X.", criteria=["What is X?"]),
            "ResearchPlan": ResearchPlan(topics=list(topics)),
            "ReportCheck": check,
            "FollowUp": triage,
        },
        tools=supervisor_model or supervisor,
    )
    reports = count(1)
    verdict = SafetyVerdict(allowed=safe, reason="" if safe else "Asks for weapon instructions.")
    standard = FakeTier(
        {"SafetyVerdict": verdict, "TopicReport": TOPIC_REPORT},
        tools=researcher,
        plain=writer or (lambda _: f"# Report v{next(reports)}"),
    )
    store = FakeStore()
    scorer = fake_scorer(evaluation or {"faithfulness": 0.9, "answer_relevancy": 0.8})
    graph = build_graph(settings, reasoning, standard, FakeSearch(), store, checkpointer=MemorySaver(), scorer=scorer)
    return DeepResearch(settings, graph=graph), store
