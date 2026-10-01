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

    async def fake_score(question, report, contexts, *, model, api_key, provider):
        captured.update(
            question=question, report=report, contexts=contexts,
            model=model, api_key=api_key, provider=provider,
        )
        return 0.87

    monkeypatch.setattr(evaluation, "_score_report_faithfulness_async", fake_score)
    evaluate = evaluation.make_faithfulness_evaluator(model="gpt-4.1-mini", api_key="secret")

    score = evaluate("What does X do?", "X does Y [S1].", ["X does Y."])

    assert score == 0.87
    assert captured == {
        "question": "What does X do?",
        "report": "X does Y [S1].",
        "contexts": ["X does Y."],
        "model": "gpt-4.1-mini",
        "api_key": "secret",
        "provider": "openai",
    }


def test_make_faithfulness_evaluator_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported faithfulness provider"):
        evaluation.make_faithfulness_evaluator(model="m", api_key="k", provider="groq")


def test_openai_judge_llm_uses_an_async_openai_client(monkeypatch):
    captured = {}

    class FakeAsyncOpenAI:
        def __init__(self, api_key):
            captured["api_key"] = api_key

    def fake_llm_factory(model, provider, client, **kwargs):
        captured.update(model=model, provider=provider, client=client, **kwargs)
        return "judge-llm"

    monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr("ragas.llms.llm_factory", fake_llm_factory)

    assert evaluation._openai_judge_llm("gpt-4.1-mini", "secret-key") == "judge-llm"
    assert captured["api_key"] == "secret-key"
    assert captured["model"] == "gpt-4.1-mini"
    assert captured["provider"] == "openai"
    assert isinstance(captured["client"], FakeAsyncOpenAI)
    assert captured["max_tokens"] == evaluation.JUDGE_MAX_TOKENS


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
        def __init__(self, client, model, provider, max_tokens):
            captured["instructor_llm_client"] = client
            captured["model"] = model
            captured["provider"] = provider
            captured["max_tokens"] = max_tokens

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
    assert captured["max_tokens"] == evaluation.JUDGE_MAX_TOKENS


REPORT = """# HTTP 429

Intro sentence.

## Scope

Covers HTTP status codes.

## Meaning

429 means too many requests [S1].

## Retry-After

Servers may send Retry-After [S2].

## Limitations

No vendor data.

## Sources

- [S1] RFC 6585 - https://example.org
"""


def test_report_sections_drop_scope_limitations_and_sources():
    sections = evaluation.report_sections(REPORT)

    assert [s.splitlines()[0] for s in sections] == ["# HTTP 429", "## Meaning", "## Retry-After"]
    assert not any("RFC 6585 - https" in s or "No vendor data" in s for s in sections)


class _Verdict:
    def __init__(self, verdict):
        self.verdict = verdict


class _Verdicts:
    def __init__(self, statements):
        self.statements = statements


def test_score_is_computed_per_section_and_in_nli_batches(monkeypatch):
    """Statements come from each scored section separately and are judged in batches of at most
    NLI_BATCH_SIZE; the score is supported / all statements across every batch."""
    import asyncio

    calls = {"sections": [], "batches": []}

    class FakeFaithfulness:
        def __init__(self, llm):
            pass

        async def _create_statements(self, question, section):
            calls["sections"].append(section.splitlines()[0])
            # 5 statements per section; the "Retry-After" section's are unsupported.
            return [f"{section.splitlines()[0]} #{i}" for i in range(5)]

        async def _create_verdicts(self, statements, context):
            calls["batches"].append(len(statements))
            return _Verdicts([_Verdict(0 if "Retry-After" in s else 1) for s in statements])

    monkeypatch.setattr("ragas.metrics.collections.Faithfulness", FakeFaithfulness)
    monkeypatch.setitem(evaluation._JUDGE_LLMS, "openai", lambda model, key: "llm")
    monkeypatch.setattr(evaluation, "NLI_BATCH_SIZE", 4)

    score = asyncio.run(evaluation._score_report_faithfulness_async(
        "q", REPORT, ["ctx"], model="m", api_key="k",
    ))

    assert sorted(calls["sections"]) == ["# HTTP 429", "## Meaning", "## Retry-After"]
    assert calls["batches"] == [4, 4, 4, 3]
    assert score == 10 / 15


def test_score_is_none_when_no_statements_are_extracted(monkeypatch):
    import asyncio

    class FakeFaithfulness:
        def __init__(self, llm):
            pass

        async def _create_statements(self, question, section):
            return []

    monkeypatch.setattr("ragas.metrics.collections.Faithfulness", FakeFaithfulness)
    monkeypatch.setitem(evaluation._JUDGE_LLMS, "openai", lambda model, key: "llm")

    assert asyncio.run(evaluation._score_report_faithfulness_async(
        "q", REPORT, ["ctx"], model="m", api_key="k",
    )) is None


def test_ragas_faithfulness_still_exposes_the_two_steps_we_call():
    """Guards the ragas upgrade path: the batched scorer calls these private steps directly."""
    from ragas.metrics.collections import Faithfulness

    assert callable(Faithfulness._create_statements)
    assert callable(Faithfulness._create_verdicts)


# ===== CONTEXT PRECISION / RESPONSE RELEVANCY =====


def test_context_precision_evaluator_returns_none_without_ranked_contexts():
    evaluate = evaluation.make_context_precision_evaluator(model="m", api_key="k")

    assert evaluate([], REPORT) is None
    assert evaluate([{"question": "q", "contexts": []}], REPORT) is None
    assert evaluate([{"question": "q", "contexts": ["c"]}], "") is None


def test_context_precision_scores_each_question_on_its_ranked_contexts(monkeypatch):
    """Each question is scored on its own contexts, in rank order, against the report body;
    the run score is the mean over questions."""
    seen = []

    class FakeContextPrecision:
        def __init__(self, llm):
            pass

        async def ascore(self, user_input, response, retrieved_contexts):
            seen.append((user_input, response, retrieved_contexts))

            class Result:
                value = {"q1": 1.0, "q2": 0.5}[user_input]
            return Result()

    monkeypatch.setattr("ragas.metrics.collections.ContextPrecisionWithoutReference",
                        FakeContextPrecision)
    monkeypatch.setitem(evaluation._JUDGE_LLMS, "openai", lambda model, key: "llm")
    evaluate = evaluation.make_context_precision_evaluator(model="m", api_key="k")

    result = evaluate([
        {"question": "q1", "contexts": ["top", "second"]},
        {"question": "q2", "contexts": ["only"]},
        {"question": "q3", "contexts": []},  # nothing retrieved: skipped
    ], REPORT)

    assert result["score"] == 0.75
    assert [(d["question"], d["contexts"], d["score"]) for d in result["per_question"]] == [
        ("q1", 2, 1.0), ("q2", 1, 0.5),
    ]
    assert sorted(user for user, _, _ in seen) == ["q1", "q2"]
    assert dict((u, c) for u, _, c in seen)["q1"] == ["top", "second"]  # rank order kept
    assert all("RFC 6585 - https" not in response for _, response, _ in seen)  # body only


def test_response_relevancy_scores_the_body_with_adapted_embeddings(monkeypatch):
    seen = {}

    class FakeLangChainEmbeddings:
        async def aembed_query(self, text):
            return [1.0, 0.0]

        async def aembed_documents(self, texts):
            return [[1.0, 0.0] for _ in texts]

    class FakeAnswerRelevancy:
        def __init__(self, llm, embeddings):
            seen["embeddings"] = embeddings

        async def ascore(self, user_input, response):
            seen.update(question=user_input, response=response)
            seen["vector"] = await self_embeddings().aembed_text(user_input)

            class Result:
                value = 0.66
            return Result()

    def self_embeddings():
        return seen["embeddings"]

    monkeypatch.setattr("ragas.metrics.collections.AnswerRelevancy", FakeAnswerRelevancy)
    monkeypatch.setitem(evaluation._JUDGE_LLMS, "openai", lambda model, key: "llm")
    evaluate = evaluation.make_response_relevancy_evaluator(
        model="m", api_key="k", embeddings=FakeLangChainEmbeddings(),
    )

    assert evaluate("What does HTTP 429 mean?", REPORT) == 0.66
    assert seen["question"] == "What does HTTP 429 mean?"
    assert "No vendor data" not in seen["response"]  # Limitations excluded
    assert seen["vector"] == [1.0, 0.0]  # LangChain embeddings reached through the adapter


def test_response_relevancy_returns_none_for_an_empty_report():
    evaluate = evaluation.make_response_relevancy_evaluator(
        model="m", api_key="k", embeddings=object(),
    )

    assert evaluate("q", "") is None


def test_ragas_exposes_the_metric_classes_we_build():
    """Guards the ragas upgrade path for the new metrics."""
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecisionWithoutReference

    assert callable(ContextPrecisionWithoutReference.ascore)
    assert callable(AnswerRelevancy.ascore)
    assert isinstance(evaluation._ragas_embeddings(object()), BaseRagasEmbedding)
