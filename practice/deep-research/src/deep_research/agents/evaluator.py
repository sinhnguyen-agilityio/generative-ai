"""Evaluator agent: scores the report for the reviewer and Langfuse. It never changes the route,
and a failure is recorded in state instead of raised."""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig

from deep_research import tracing
from deep_research.agents.utils import thread_id
from deep_research.config import Settings
from deep_research.prompts import EVALUATION_PROMPT
from deep_research.state import Evaluation, ResearchState

logger = logging.getLogger(__name__)


class EvaluatorAgent:
    def __init__(self, llm, settings: Settings):
        self.llm = llm.structured(Evaluation)
        self.langfuse = settings.langfuse_enabled

    def evaluate(self, state: ResearchState, config: RunnableConfig):
        prompt = EVALUATION_PROMPT.format(
            brief=state["brief"], notes="\n\n---\n\n".join(state["notes"]), report=state["report"]
        )
        try:
            result = self.llm.invoke(prompt)
        except Exception as exc:
            logger.exception("Evaluation failed")
            return {"evaluation": {"error": f"{type(exc).__name__}: {exc}"}}
        for name in ("faithfulness", "relevance"):
            tracing.score(self.langfuse, thread_id(config), name, getattr(result, name), result.comment)
        return {"evaluation": result.model_dump()}
