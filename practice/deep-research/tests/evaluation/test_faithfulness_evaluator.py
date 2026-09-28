import pytest

from intelligence_collector.observability import evaluation


def test_make_faithfulness_evaluator_returns_none_for_empty_report():
    evaluate = evaluation.make_faithfulness_evaluator(model="gemini-3.8-flash", api_key="k")

    assert evaluate("question", "", ["some context"]) is None


def test_make_faithfulness_evaluator_returns_none_for_no_contexts():
    evaluate = evaluation.make_faithfulness_evaluator(model="gemini-3.8-flash", api_key="k")

    assert evaluate("question", "a report", []) is None


def test_make_faithfulness_evaluator_calls_the_async_scorer_with_the_right_arguments(monkeypatch):
    captured = {}

    async def fake_score(question, report, contexts, *, model, api_key):
        captured.update(question=question, report=report, contexts=contexts, model=model, api_key=api_key)
        return 0.87

    monkeypatch.setattr(evaluation, "_score_report_faithfulness_async", fake_score)
    evaluate = evaluation.make_faithfulness_evaluator(model="gemini-3.8-flash", api_key="secret")

    score = evaluate("What does X do?", "X does Y [S1].", ["X does Y."])

    assert score == 0.87
    assert captured == {
        "question": "What does X do?",
        "report": "X does Y [S1].",
        "contexts": ["X does Y."],
        "model": "gemini-3.8-flash",
        "api_key": "secret",
    }


def test_make_faithfulness_evaluator_propagates_judge_llm_failures(monkeypatch):
    async def failing_score(*args, **kwargs):
        raise RuntimeError("429 rate limited")

    monkeypatch.setattr(evaluation, "_score_report_faithfulness_async", failing_score)
    evaluate = evaluation.make_faithfulness_evaluator(model="gemini-3.8-flash", api_key="k")

    with pytest.raises(RuntimeError, match="rate limited"):
        evaluate("q", "report text", ["context"])
