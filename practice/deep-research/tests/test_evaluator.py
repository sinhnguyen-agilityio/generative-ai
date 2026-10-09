"""The evaluator scores the final report against the passages behind the cited sources."""

from deep_research.agents.evaluator import EvaluatorAgent
from deep_research.model_spec import ModelSpec
from deep_research.settings import Settings
from deep_research.utils.evaluation import _ragas_llm

SETTINGS = Settings(reasoning=ModelSpec("fake"), standard=ModelSpec("fake"))
CONFIG = {"configurable": {"thread_id": "t"}}


def note(*passages):
    """Build a topic note containing the supplied source passages.

    Args:
        passages: Evidence strings attached to one cited source.

    Returns:
        Minimal topic note for evaluation tests.
    """
    return {"report": "## Status: complete", "sources": [{"id": 1, "title": "T", "url": "u", "full_text": True, "passages": list(passages)}]}


def test_the_scorer_gets_the_report_without_sources_and_every_cited_passage_once():
    """Verify that the scorer gets the report without sources and every cited passage once."""
    calls = []

    def scorer(brief, report, contexts):
        """Record cleaned scoring inputs and return fixed metrics."""
        calls.append((brief, report, contexts))
        return {"faithfulness": 0.8, "answer_relevancy": 0.7}

    state = {
        "brief": "Compare costs.",
        "report": "# Costs\nGemini is $0.30 [1][2].\n\n## Sources\n[1] Gemini pricing: https://ai.google.dev/pricing",
        "notes": [note("p1", "p2"), note("p2", "p3"), {"report": "## Status: failed", "sources": []}],
    }
    result = EvaluatorAgent(SETTINGS, scorer).evaluate(state, CONFIG)
    assert result == {"evaluation": {"faithfulness": 0.8, "answer_relevancy": 0.7}}
    assert calls == [("Compare costs.", "# Costs\nGemini is $0.30 .", ["p1", "p2", "p3"])]


def test_a_scorer_failure_is_recorded_not_raised():
    """Verify that a scorer failure is recorded not raised."""
    def scorer(brief, report, contexts):
        """Simulate an unavailable evaluator to exercise error recording."""
        raise RuntimeError("ragas down")

    result = EvaluatorAgent(SETTINGS, scorer).evaluate({"brief": "b", "report": "r", "notes": []}, CONFIG)
    assert result == {"evaluation": {"error": "RuntimeError: ragas down"}}


def test_the_ragas_judge_is_an_async_gemini_client(monkeypatch):
    """Verify judge construction uses async Gemini without making a live API request."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")  # the client is built offline; nothing is sent
    llm = _ragas_llm("gemini-test")
    assert (llm.model, llm.provider, llm.is_async) == ("gemini-test", "google", True)
