"""The evaluator scores the final report against the passages behind the cited sources."""

from deep_research.agents.evaluator import EvaluatorAgent
from deep_research.config import ModelSpec, Settings

SETTINGS = Settings(reasoning=ModelSpec("fake"), standard=ModelSpec("fake"))
CONFIG = {"configurable": {"thread_id": "t"}}


def note(*passages):
    return {"report": "## Status: complete", "sources": [{"id": 1, "title": "T", "url": "u", "full_text": True, "passages": list(passages)}]}


def test_the_scorer_gets_the_report_without_sources_and_every_cited_passage_once():
    calls = []

    def scorer(brief, report, contexts):
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
    def scorer(brief, report, contexts):
        raise RuntimeError("ragas down")

    result = EvaluatorAgent(SETTINGS, scorer).evaluate({"brief": "b", "report": "r", "notes": []}, CONFIG)
    assert result == {"evaluation": {"error": "RuntimeError: ragas down"}}


def test_ragas_needs_an_openai_model():
    groq = Settings(reasoning=ModelSpec("openai/gpt-oss-120b", provider="groq"), standard=ModelSpec("fake"))
    try:
        EvaluatorAgent(groq)
    except ValueError as exc:
        assert "openai provider only" in str(exc)
    else:
        raise AssertionError("expected ValueError")
