"""Evaluator agent: scores the final report with RAGAS for the reviewer and Langfuse. It never
changes the route, and a failure is recorded in state instead of raised.

- faithfulness: share of the report's claims that the source passages the researchers read support.
- answer_relevancy: how directly the report answers the brief.

The contexts are the passages behind the sources the topic notes cite, so faithfulness measures the
whole pipeline, not only whether the writer copied the notes.

The judge is a Gemini model (``evaluation_model``) through the google-genai SDK, apart from the
research tiers, so the models that write the report do not grade it."""

from __future__ import annotations

import asyncio
import logging

from langchain_core.runnables import RunnableConfig

from deep_research import tracing
from deep_research.agents.utils import CITATION, passages_by_url, thread_id
from deep_research.agents.writer import SOURCES_HEADING
from deep_research.config import Settings
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)

TIMEOUT_MS = 120_000


class EvaluatorAgent:
    def __init__(self, settings: Settings, scorer=None):
        """``scorer(brief, report, contexts) -> {name: score}``; RAGAS on ``evaluation_model`` by default."""
        self.scorer = scorer or RagasScorer(settings.evaluation_model, settings.embedding_model, settings.embedding_provider)
        self.langfuse = settings.langfuse_enabled

    def evaluate(self, state: ResearchState, config: RunnableConfig):
        report = CITATION.sub("", state["report"].partition(SOURCES_HEADING)[0]).strip()
        try:
            scores = self.scorer(state["brief"], report, contexts(state["notes"]))
        except Exception as exc:
            logger.exception("Evaluation failed")
            return {"evaluation": {"error": f"{type(exc).__name__}: {exc}"}}
        for name, value in scores.items():
            tracing.score(self.langfuse, thread_id(config), name, value)
        return {"evaluation": scores}


def contexts(notes: list[dict]) -> list[str]:
    """The passages of every source the topic notes cite, each passage once."""
    return list(dict.fromkeys(p for texts in passages_by_url(notes).values() for p in texts))


class RagasScorer:
    def __init__(self, model: str, embedding_model: str, embedding_provider: str):
        from deep_research.search import _embeddings

        self.model = model
        self.embeddings = _embeddings(embedding_model, embedding_provider)

    def __call__(self, brief: str, report: str, contexts: list[str]) -> dict[str, float]:
        return asyncio.run(self._score(brief, report, contexts))

    async def _score(self, brief: str, report: str, contexts: list[str]) -> dict[str, float]:
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness

        # A new client per run: an async client cannot be reused across asyncio.run loops.
        llm = _ragas_llm(self.model)
        faithfulness, relevancy = await asyncio.gather(
            Faithfulness(llm=llm).ascore(user_input=brief, response=report, retrieved_contexts=contexts),
            AnswerRelevancy(llm=llm, embeddings=_ragas_embeddings(self.embeddings)).ascore(
                user_input=brief, response=report
            ),
        )
        return {"faithfulness": float(faithfulness.value), "answer_relevancy": float(relevancy.value)}


def _ragas_llm(model: str):
    """A Gemini judge on an async google-genai client (reads GOOGLE_API_KEY).

    ragas 0.4.3's ``llm_factory`` wraps a google-genai client for sync calls only, and the metrics
    call ``agenerate``, so the async Instructor client is built here. Instructor drops ragas'
    1024-token output cap for Gemini, so a long claim list is not truncated."""
    import instructor
    from google import genai
    from google.genai import types
    from ragas.llms.base import InstructorLLM

    client = genai.Client(http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    return InstructorLLM(client=instructor.from_genai(client, use_async=True), model=model, provider="google")


def _ragas_embeddings(embeddings):
    """The project's LangChain embeddings behind the interface ragas' collection metrics expect."""
    from ragas.embeddings import BaseRagasEmbedding

    class Embeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs) -> list[float]:
            return embeddings.embed_query(text)

        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            return await embeddings.aembed_query(text)

    return Embeddings()
