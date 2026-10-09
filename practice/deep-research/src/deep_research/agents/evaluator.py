"""Evaluator agent: scores the final report with RAGAS for the reviewer and Langfuse. It never
changes the route, and a failure is recorded in state instead of raised.

- faithfulness: share of the report's claims that the source passages the researchers read support.
- answer_relevancy: how directly the report answers the brief.

The contexts are the passages behind the sources the topic notes cite, so faithfulness measures the
whole pipeline, not only whether the writer copied the notes.

The judge is a Gemini model (``evaluation_model``) through the google-genai SDK, apart from the
research tiers, so the models that write the report do not grade it."""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig

from evaluation.ragas_scorer import RagasScorer
from config.settings import Settings
from deep_research.states.research_state import ResearchState
from deep_research.utils import tracing
from deep_research.utils.notes import contexts
from deep_research.utils.report import SOURCES_HEADING
from deep_research.utils.runtime import thread_id
from deep_research.utils.text import CITATION

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    def __init__(self, settings: Settings, scorer=None):
        """Configure report scoring and optional metric tracing.

        Args:
            settings: Judge/embedding model settings and tracing flag.
            scorer: Optional callable (brief, report, contexts) returning named scores; defaults to RAGAS.
        """
        self.scorer = scorer or RagasScorer(settings.evaluation_model, settings.embedding_model, settings.embedding_provider)
        self.langfuse = settings.langfuse_enabled

    def evaluate(self, state: ResearchState, config: RunnableConfig):
        """Score the report without its citation markers/source list and record evaluation errors.

        Args:
            state: Brief, report, and topic notes carrying source passages.
            config: Graph configuration used to attach scores to the research thread.

        Returns:
            Evaluation scores or an error dictionary; evaluation does not change routing.
        """
        report = CITATION.sub("", state["report"].partition(SOURCES_HEADING)[0]).strip()
        try:
            scores = self.scorer(state["brief"], report, contexts(state["notes"]))
        except Exception as exc:
            logger.exception("Evaluation failed")
            return {"evaluation": {"error": f"{type(exc).__name__}: {exc}"}}
        for name, value in scores.items():
            tracing.score(self.langfuse, thread_id(config), name, value)
        return {"evaluation": scores}
