from langchain_core.runnables import (
    RunnableParallel,
    RunnableLambda,
)

from models.analysis_report import AnalysisReport
from chains import (
    urgency,
    escalation,
    reply,
)

from tracing.client import langfuse_handler
from langfuse import get_client

langfuse = get_client()


class ResponseAnalysis:
    def __init__(self):
        self._workflow = (
            RunnableParallel(
                urgency=urgency.build(),
                escalation=escalation.build(),
                suggested_reply=reply.build(),
                report=RunnableLambda(lambda report: report),
            )
            |
            RunnableLambda(self._merge)
        )

    def build(self):

        chain = self._workflow

        def invoke(report):
            with langfuse.start_as_current_observation(
                name="response_analysis",
                as_type="span"
            ):
                return chain.invoke(
                    report,
                    config={
                        "callbacks": [langfuse_handler]
                    }
                )

        return RunnableLambda(invoke)

    @staticmethod
    def _merge(result: dict) -> AnalysisReport:
        report: AnalysisReport = result["report"]
        return report.model_copy(
            update={
                "urgency": result["urgency"],
                "escalation": result["escalation"],
                "suggested_reply": result["suggested_reply"],
            }
        )
