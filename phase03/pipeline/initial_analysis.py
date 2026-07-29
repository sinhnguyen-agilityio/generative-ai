from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
)
from chains import summary, intent, sentiment, entity
from models.analysis_report import AnalysisReport

from tracing.client import langfuse_handler
from langfuse import get_client


langfuse = get_client()


class InitialAnalysis:
    def __init__(self):
        self._workflow = (
            RunnableParallel(
                summary=summary.build(),
                intent=intent.build(),
                sentiment=sentiment.build(),
                entities=entity.build(),
            )
            |
            RunnableLambda(self._build_report)
        )

    def _build_report(self, result):
        return AnalysisReport(
            summary=result["summary"],
            intent=result["intent"],
            sentiment=result["sentiment"],
            entities=result["entities"],
        )

    def invoke(self, ticket):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="initial_analysis"
        ):
            return self._workflow.invoke(
                ticket,
                config={
                    "callbacks": [langfuse_handler]
                }
            )

    def batch(self, tickets: list,) -> list[AnalysisReport]:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="initial_analysis_batch",
        ):
            return self._workflow.batch(
                tickets,
                config={
                    "callbacks": [langfuse_handler]
                }
            )
