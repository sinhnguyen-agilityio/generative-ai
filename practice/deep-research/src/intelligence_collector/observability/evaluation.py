from __future__ import annotations

import asyncio
from typing import Callable

from intelligence_collector.observability._ragas_compat import ensure_ragas_importable

ensure_ragas_importable()


def faithfulness_score(answer: str, contexts: list[str]) -> float:
    """Deterministic faithfulness baseline used by the repeatable local evaluation."""
    if not answer.strip() or not contexts:
        return 0.0
    words = {word.strip(".,:;!?()[]").lower() for word in answer.split() if len(word) > 3}
    supported = {word.strip(".,:;!?()[]").lower() for text in contexts for word in text.split()}
    return round(len(words & supported) / max(len(words), 1), 4)


FaithfulnessEvaluator = Callable[[str, str, list[str]], "float | None"]


def _google_judge_llm(model: str, api_key: str):
    """Build a ragas judge LLM around the Gemini API.

    ``ragas.llms.llm_factory``'s own Google/instructor path always patches the client
    *synchronously* (``instructor.from_genai(client)``, no ``use_async``), which breaks
    RAGAS's async ``Faithfulness.ascore()`` with ``TypeError: Cannot use agenerate() with a
    synchronous client`` (verified against the real API). Patch with ``use_async=True``
    directly instead of going through ``llm_factory``.
    """
    import instructor
    from google import genai
    from ragas.llms.base import InstructorLLM

    client = genai.Client(api_key=api_key)
    patched = instructor.from_genai(client, use_async=True)
    return InstructorLLM(client=patched, model=model, provider="google")


async def _score_report_faithfulness_async(
    question: str, report: str, contexts: list[str], *, model: str, api_key: str
) -> float:
    from ragas.metrics.collections import Faithfulness

    scorer = Faithfulness(llm=_google_judge_llm(model, api_key))
    result = await scorer.ascore(
        user_input=question, response=report, retrieved_contexts=contexts
    )
    return float(result.value)


def make_faithfulness_evaluator(*, model: str, api_key: str) -> FaithfulnessEvaluator:
    """Return a ``(question, report, contexts) -> score`` callable that scores an entire
    report's RAGAS Faithfulness with the given Gemini model as judge.

    ``contexts`` should be the excerpt text of every source the report actually cites (see
    ``source_verification.cited_source_excerpts``) -- what the report was supposed to stay
    faithful to. Returns ``None`` when there is nothing to score; raises on a judge-LLM
    failure (rate limit, transient API error) so the caller decides how to record it. Per
    the "Observability and evaluation" design decision (research.md), this score MUST stay
    informational and never gate verification or publication.
    """

    def evaluate(question: str, report: str, contexts: list[str]) -> float | None:
        if not report.strip() or not contexts:
            return None
        return asyncio.run(
            _score_report_faithfulness_async(
                question, report, contexts, model=model, api_key=api_key
            )
        )

    return evaluate
