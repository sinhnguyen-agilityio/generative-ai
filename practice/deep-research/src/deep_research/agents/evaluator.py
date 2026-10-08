"""Evaluator agent: scores the final report with RAGAS for the reviewer and Langfuse. It never
changes the route, and a failure is recorded in state instead of raised.

- faithfulness: share of the report's claims that the source passages the researchers read support.
- answer_relevancy: how directly the report answers the brief.

The contexts are the passages behind the sources the topic notes cite, so faithfulness measures the
whole pipeline, not only whether the writer copied the notes."""

from __future__ import annotations

import asyncio
import logging

from langchain_core.runnables import RunnableConfig

from deep_research import tracing
from deep_research.agents.utils import CITATION, thread_id
from deep_research.agents.writer import SOURCES_HEADING
from deep_research.config import ModelSpec, Settings
from deep_research.state import ResearchState

logger = logging.getLogger(__name__)

# Faithfulness lists every claim of the report with a verdict and a reason; a full report needs room.
MAX_COMPLETION_TOKENS = 32768


class EvaluatorAgent:
    def __init__(self, settings: Settings, scorer=None):
        """``scorer(brief, report, contexts) -> {name: score}``; RAGAS on the reasoning tier by default."""
        self.scorer = scorer or RagasScorer(settings.reasoning, settings.embedding_model, settings.embedding_provider)
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
    passages = {}
    for note in notes:
        for source in note["sources"]:
            for passage in source.get("passages", []):
                passages.setdefault(passage, None)
    return list(passages)


class RagasScorer:
    def __init__(self, spec: ModelSpec, embedding_model: str, embedding_provider: str):
        if (spec.provider or "openai") != "openai":
            raise ValueError(f"RAGAS evaluation supports the openai provider only, not {spec.provider}")
        from deep_research.search import _embeddings

        self.spec = spec
        self.embeddings = _embeddings(embedding_model, embedding_provider)

    def __call__(self, brief: str, report: str, contexts: list[str]) -> dict[str, float]:
        return asyncio.run(self._score(brief, report, contexts))

    async def _score(self, brief: str, report: str, contexts: list[str]) -> dict[str, float]:
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness

        # A new client per run: an async client cannot be reused across asyncio.run loops.
        llm = _ragas_llm(self.spec)
        faithfulness, relevancy = await asyncio.gather(
            Faithfulness(llm=llm).ascore(user_input=brief, response=report, retrieved_contexts=contexts),
            AnswerRelevancy(llm=llm, embeddings=_ragas_embeddings(self.embeddings)).ascore(
                user_input=brief, response=report
            ),
        )
        return {"faithfulness": float(faithfulness.value), "answer_relevancy": float(relevancy.value)}


def _ragas_llm(spec: ModelSpec):
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    llm = llm_factory(spec.model, provider="openai", client=AsyncOpenAI(timeout=spec.timeout))

    # ragas 0.4.3 reads "gpt-5.4-nano" as version int("5.4"), fails, and keeps max_tokens, which
    # GPT-5 models reject. Map the parameters as ragas does for GPT-5 models; drop this once fixed.
    if spec.model.startswith("gpt-5"):
        def openai_params():
            args = dict(llm.model_args)
            args["max_completion_tokens"] = max(args.pop("max_tokens", 0) or 0, MAX_COMPLETION_TOKENS)
            args["temperature"] = 1.0
            args.pop("top_p", None)
            if spec.kwargs.get("reasoning_effort"):
                args["reasoning_effort"] = spec.kwargs["reasoning_effort"]
            return args

        llm._map_openai_params = openai_params
    return llm


def _ragas_embeddings(embeddings):
    """The project's LangChain embeddings behind the interface ragas' collection metrics expect."""
    from ragas.embeddings import BaseRagasEmbedding

    class Embeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs) -> list[float]:
            return embeddings.embed_query(text)

        async def aembed_text(self, text: str, **kwargs) -> list[float]:
            return await embeddings.aembed_query(text)

    return Embeddings()
