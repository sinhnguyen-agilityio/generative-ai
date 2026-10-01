"""RAGAS scoring of the final report: Faithfulness, Context Precision and Response Relevancy.
Observability only: a score never gates verification or publication.
"""

from __future__ import annotations

import asyncio
import re
from typing import Callable

from intelligence_collector.observability._ragas_compat import ensure_ragas_importable

ensure_ragas_importable()

# Output cap per judge call. RAGAS defaults to 1024, which a whole report's statement list or
# verdicts overflow ("The output is incomplete due to a max_tokens length limit").
JUDGE_MAX_TOKENS = 4096
# Statements judged per NLI call; each verdict carries a reason, so this bounds the output size.
NLI_BATCH_SIZE = 12
# Judge calls in flight at once (statement extraction and NLI batches share the limit).
JUDGE_CONCURRENCY = 4
# Sections that make no evidence-backed claims (report guidelines exempt them from citations).
UNSCORED_SECTIONS = {"scope", "limitations", "sources"}
_HEADING = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def report_sections(report: str) -> list[str]:
    """Split a Markdown report into its ``## `` sections (text before the first one included) and
    drop the ones in ``UNSCORED_SECTIONS``, which only restate the scope, disclose gaps, or list
    sources and so have nothing to be faithful to."""
    starts = [match.start() for match in _HEADING.finditer(report)]
    chunks = [report[a:b] for a, b in zip([0, *starts], [*starts, len(report)])]
    sections = []
    for chunk in chunks:
        heading = _HEADING.match(chunk)
        if heading and heading.group(1).strip().lower() in UNSCORED_SECTIONS:
            continue
        if chunk.strip():
            sections.append(chunk.strip())
    return sections


def report_body(report: str) -> str:
    """The report's scored sections (see ``report_sections``) joined back into one text."""
    return "\n\n".join(report_sections(report))


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
    return InstructorLLM(client=patched, model=model, provider="google", max_tokens=JUDGE_MAX_TOKENS)


def _openai_judge_llm(model: str, api_key: str):
    """Build a ragas judge LLM around the OpenAI API. An ``AsyncOpenAI`` client keeps
    ``Faithfulness.ascore()`` on a native async path (no sync-client patching needed)."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    return llm_factory(
        model, provider="openai", client=AsyncOpenAI(api_key=api_key), max_tokens=JUDGE_MAX_TOKENS,
    )


_JUDGE_LLMS = {"openai": _openai_judge_llm, "google_genai": _google_judge_llm}


async def _score_report_faithfulness_async(
    question: str, report: str, contexts: list[str], *, model: str, api_key: str,
    provider: str = "openai",
) -> float | None:
    """RAGAS Faithfulness over the report's scored sections: supported statements / all
    statements. Same two steps as ``Faithfulness.ascore`` (extract statements, then NLI verdicts
    against the contexts), but per section and in NLI batches, so no single judge call has to
    emit a whole report's worth of output. ``None`` when no statements were extracted."""
    from ragas.metrics.collections import Faithfulness

    scorer = Faithfulness(llm=_JUDGE_LLMS[provider](model, api_key))
    context = "\n".join(contexts)
    limit = asyncio.Semaphore(JUDGE_CONCURRENCY)

    # `_create_statements` / `_create_verdicts` are the two steps `Faithfulness.ascore` runs on
    # the whole response (ragas 0.4.x); calling them directly lets each call stay small.
    async def statements_of(section: str) -> list[str]:
        async with limit:
            return await scorer._create_statements(question, section)

    async def verdicts_of(batch: list[str]) -> list[int]:
        async with limit:
            result = await scorer._create_verdicts(batch, context)
        return [1 if item.verdict else 0 for item in result.statements]

    per_section = await asyncio.gather(*(statements_of(s) for s in report_sections(report)))
    statements = [statement for found in per_section for statement in found]
    if not statements:
        return None
    batches = [statements[i:i + NLI_BATCH_SIZE] for i in range(0, len(statements), NLI_BATCH_SIZE)]
    verdicts = [v for batch in await asyncio.gather(*(verdicts_of(b) for b in batches)) for v in batch]
    if not verdicts:
        return None
    return sum(verdicts) / len(verdicts)


def make_faithfulness_evaluator(
    *, model: str, api_key: str, provider: str = "openai",
) -> FaithfulnessEvaluator:
    """Return a ``(question, report, contexts) -> score`` callable that scores a report's RAGAS
    Faithfulness (its body sections; see ``report_sections``) with the given judge model:
    ``provider`` is ``"openai"``
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


# ===== CONTEXT PRECISION =====

# One research question's retrieval as the report writer saw it: {"question": str,
# "contexts": [excerpt, ...]} with the excerpts in retrieval rank order.
ContextPrecisionEvaluator = Callable[[list[dict], str], "dict | None"]


async def _score_context_precision_async(
    questions: list[dict], report: str, *, model: str, api_key: str, provider: str = "openai",
) -> dict:
    """RAGAS Context Precision (without reference) per research question: one judge verdict per
    retrieved context ("was it useful for this report?"), then rank-weighted average precision.
    The run score is the mean over questions."""
    from ragas.metrics.collections import ContextPrecisionWithoutReference

    scorer = ContextPrecisionWithoutReference(llm=_JUDGE_LLMS[provider](model, api_key))
    limit = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def score(item: dict) -> float:
        async with limit:
            result = await scorer.ascore(
                user_input=item["question"], response=report, retrieved_contexts=item["contexts"],
            )
        return float(result.value)

    scores = await asyncio.gather(*(score(item) for item in questions))
    return {
        "score": sum(scores) / len(scores),
        "per_question": [
            {"question": item["question"], "contexts": len(item["contexts"]), "score": value}
            for item, value in zip(questions, scores)
        ],
    }


def make_context_precision_evaluator(
    *, model: str, api_key: str, provider: str = "openai",
) -> ContextPrecisionEvaluator:
    """Return a ``(questions, report) -> {"score", "per_question"}`` callable scoring whether the
    retrieval ranked the sources that the report needed at the top. ``questions`` is the writer's
    per-question ranked evidence (see ``BVFNodes.synthesizing``). Returns ``None`` when there is
    nothing to score; raises on a judge-LLM failure. Informational only."""
    if provider not in _JUDGE_LLMS:
        raise ValueError(f"Unsupported judge provider {provider!r}; use one of {sorted(_JUDGE_LLMS)}")

    def evaluate(questions: list[dict], report: str) -> dict | None:
        """Score one report's retrieval; ``None`` when there is nothing to score."""
        body = report_body(report)
        scored = [item for item in questions if item.get("question") and item.get("contexts")]
        if not body or not scored:
            return None
        return asyncio.run(_score_context_precision_async(
            scored, body, model=model, api_key=api_key, provider=provider,
        ))

    return evaluate


# ===== RESPONSE RELEVANCY =====

ResponseRelevancyEvaluator = Callable[[str, str], "float | None"]


def _ragas_embeddings(embeddings):
    """Adapt a LangChain ``Embeddings`` (the one the vector index uses) to RAGAS's modern
    embedding interface, which collections metrics require."""
    from ragas.embeddings.base import BaseRagasEmbedding

    class LangChainEmbedding(BaseRagasEmbedding):
        def embed_text(self, text, **kwargs):
            return embeddings.embed_query(text)

        async def aembed_text(self, text, **kwargs):
            return await embeddings.aembed_query(text)

        def embed_texts(self, texts, **kwargs):
            return embeddings.embed_documents(list(texts))

        async def aembed_texts(self, texts, **kwargs):
            return await embeddings.aembed_documents(list(texts))

    return LangChainEmbedding()


async def _score_response_relevancy_async(
    question: str, report: str, *, model: str, api_key: str, provider: str, embeddings,
) -> float:
    """RAGAS Response Relevancy: the judge writes questions the report answers; the score is
    their mean cosine similarity to the research question (0 if the report is noncommittal)."""
    from ragas.metrics.collections import AnswerRelevancy

    scorer = AnswerRelevancy(
        llm=_JUDGE_LLMS[provider](model, api_key), embeddings=_ragas_embeddings(embeddings),
    )
    result = await scorer.ascore(user_input=question, response=report)
    return float(result.value)


def make_response_relevancy_evaluator(
    *, model: str, api_key: str, embeddings, provider: str = "openai",
) -> ResponseRelevancyEvaluator:
    """Return a ``(question, report) -> score`` callable scoring how directly the report body
    addresses the research question. ``embeddings`` is a LangChain ``Embeddings``. Scoring the
    body only matters here: a ``## Limitations`` section disclosing gaps could otherwise get the
    whole report judged noncommittal (score 0). Returns ``None`` when there is nothing to score;
    raises on a judge or embedding failure. Informational only."""
    if provider not in _JUDGE_LLMS:
        raise ValueError(f"Unsupported judge provider {provider!r}; use one of {sorted(_JUDGE_LLMS)}")

    def evaluate(question: str, report: str) -> float | None:
        """Score one report; ``None`` when there is nothing to score."""
        body = report_body(report)
        if not body or not question.strip():
            return None
        return asyncio.run(_score_response_relevancy_async(
            question, body, model=model, api_key=api_key, provider=provider, embeddings=embeddings,
        ))

    return evaluate
