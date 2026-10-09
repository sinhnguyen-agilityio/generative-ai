"""Ragas scorer definitions."""

from __future__ import annotations

import asyncio

from deep_research.utils.evaluation import _ragas_embeddings, _ragas_llm


class RagasScorer:
    def __init__(self, model: str, embedding_model: str, embedding_provider: str):
        """Configure the Gemini judge model and LangChain embedding client.

        Args:
            model: Gemini model used to judge the report.
            embedding_model: Embedding model used for answer relevancy.
            embedding_provider: Provider selecting the embedding client.
        """
        from deep_research.utils.embeddings import _embeddings

        self.model = model
        self.embeddings = _embeddings(embedding_model, embedding_provider)

    def __call__(self, brief: str, report: str, contexts: list[str]) -> dict[str, float]:
        """Run asynchronous report scoring through the synchronous scorer interface.

        Args:
            brief: Research question represented by the brief.
            report: Report body stripped of generated source lists and citation markers.
            contexts: Source evidence used to judge faithfulness.

        Returns:
            Faithfulness and answer_relevancy scores.
        """
        return asyncio.run(self._score(brief, report, contexts))

    async def _score(self, brief: str, report: str, contexts: list[str]) -> dict[str, float]:
        """Compute faithfulness and answer relevancy concurrently with a fresh async judge.

        Args:
            brief: Research brief supplied as user input to both metrics.
            report: Report body being evaluated.
            contexts: Retrieved evidence used for the faithfulness metric.

        Returns:
            Numeric faithfulness and answer_relevancy scores.
        """
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
