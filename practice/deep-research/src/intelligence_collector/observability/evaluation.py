"""RAGAS Faithfulness scoring of the final report. Observability only: a score never gates
verification or publication.
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
