"""RAGAS Faithfulness scoring of the final report. Observability only: a score never gates
verification or publication.
"""

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


def _openai_judge_llm(model: str, api_key: str):
    """Build a ragas judge LLM around the OpenAI API. An ``AsyncOpenAI`` client keeps
    ``Faithfulness.ascore()`` on a native async path (no sync-client patching needed)."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    return llm_factory(model, provider="openai", client=AsyncOpenAI(api_key=api_key))


_JUDGE_LLMS = {"openai": _openai_judge_llm, "google_genai": _google_judge_llm}


async def _score_report_faithfulness_async(
    question: str, report: str, contexts: list[str], *, model: str, api_key: str,
    provider: str = "openai",
) -> float:
    from ragas.metrics.collections import Faithfulness

    scorer = Faithfulness(llm=_JUDGE_LLMS[provider](model, api_key))
    result = await scorer.ascore(
        user_input=question, response=report, retrieved_contexts=contexts
    )
    return float(result.value)


def make_faithfulness_evaluator(
    *, model: str, api_key: str, provider: str = "openai",
) -> FaithfulnessEvaluator:
    """Return a ``(question, report, contexts) -> score`` callable that scores an entire
    report's RAGAS Faithfulness with the given judge model: ``provider`` is ``"openai"``
    (e.g. ``gpt-4.1-mini``, the default) or ``"google_genai"`` (e.g. ``gemini-3.8-flash``).

    ``contexts`` should be the excerpt text of every source the report actually cites (see
    ``source_verification.cited_source_excerpts``) -- what the report was supposed to stay
    faithful to. Returns ``None`` when there is nothing to score; raises on a judge-LLM
    failure (rate limit, transient API error) so the caller decides how to record it. Per
    the "Observability and evaluation" design decision (research.md), this score MUST stay
    informational and never gate verification or publication.
    """

    if provider not in _JUDGE_LLMS:
        raise ValueError(f"Unsupported faithfulness provider {provider!r}; use one of {sorted(_JUDGE_LLMS)}")

    def evaluate(question: str, report: str, contexts: list[str]) -> float | None:
        """Score one report; ``None`` when there is nothing to score."""
        if not report.strip() or not contexts:
            return None
        return asyncio.run(
            _score_report_faithfulness_async(
                question, report, contexts, model=model, api_key=api_key, provider=provider,
            )
        )

    return evaluate
