import pytest

from intelligence_collector.observability import evaluation


def test_make_faithfulness_evaluator_returns_none_for_empty_report():
    evaluate = evaluation.make_faithfulness_evaluator(model="gemini-3.7-flash", api_key="k")

    assert evaluate("question", "", ["some context"]) is None


def test_make_faithfulness_evaluator_returns_none_for_no_contexts():
    evaluate = evaluation.make_faithfulness_evaluator(model="gemini-3.7-flash", api_key="k")

    assert evaluate("question", "a report", []) is None


def test_make_faithfulness_evaluator_calls_the_async_scorer_with_the_right_arguments(monkeypatch):
    captured = {}

    async def fake_score(question, report, contexts, *, model, api_key):
        captured.update(question=question, report=report, contexts=contexts, model=model, api_key=api_key)
        return 0.87

    monkeypatch.setattr(evaluation, "_score_report_faithfulness_async", fake_score)
    evaluate = evaluation.make_faithfulness_evaluator(model="gemini-3.7-flash", api_key="secret")

    score = evaluate("What does X do?", "X does Y [S1].", ["X does Y."])

    assert score == 0.87
    assert captured == {
        "question": "What does X do?",
        "report": "X does Y [S1].",
        "contexts": ["X does Y."],
        "model": "gemini-3.7-flash",
        "api_key": "secret",
    }


def test_make_faithfulness_evaluator_propagates_judge_llm_failures(monkeypatch):
    async def failing_score(*args, **kwargs):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(evaluation, "_score_report_faithfulness_async", failing_score)
    evaluate = evaluation.make_faithfulness_evaluator(model="gemini-3.7-flash", api_key="k")

    with pytest.raises(RuntimeError, match="rate limited"):
        evaluate("q", "report text", ["context"])


def test_google_judge_llm_patches_the_client_for_async_use(monkeypatch):
    """ragas.llms.llm_factory's own Google/instructor path always patches the client
    *synchronously* (instructor.from_genai(client), no use_async), which breaks RAGAS's async
    Faithfulness.ascore() with a real TypeError (verified against the live API). So this patches
    with use_async=True directly instead of going through llm_factory."""
    captured = {}

    class FakeGenaiClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key

    def fake_from_genai(client, use_async):
        captured["client"] = client
        captured["use_async"] = use_async
        return "patched-client"

    class FakeInstructorLLM:
        def __init__(self, client, model, provider):
            captured["instructor_llm_client"] = client
            captured["model"] = model
            captured["provider"] = provider

        def __eq__(self, other):
            return isinstance(other, FakeInstructorLLM)

    monkeypatch.setattr("google.genai.Client", FakeGenaiClient)
    monkeypatch.setattr("instructor.from_genai", fake_from_genai)
    monkeypatch.setattr("ragas.llms.base.InstructorLLM", FakeInstructorLLM)

    result = evaluation._google_judge_llm("gemini-3.7-flash", "secret-key")

    assert isinstance(result, FakeInstructorLLM)
    assert captured["api_key"] == "secret-key"
    assert isinstance(captured["client"], FakeGenaiClient)
    assert captured["use_async"] is True
    assert captured["model"] == "gemini-3.7-flash"
    assert captured["provider"] == "google"
    assert captured["instructor_llm_client"] == "patched-client"
