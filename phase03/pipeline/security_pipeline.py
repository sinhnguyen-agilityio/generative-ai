from langchain_core.runnables import RunnableLambda

from middlewares.toxicity import OpenAIModerationDetector
from models.ticket import SecureTicket
from middlewares.pii import PIIDetector
from config import settings
from tracing.client import langfuse_handler
from langfuse import get_client


langfuse = get_client()


class SecurityPipeline:
    def __init__(self):
        self._pii_detector = PIIDetector()
        self._moderation = OpenAIModerationDetector(
            api_key=settings.openai_api_key)
        self._chain = (
            RunnableLambda(self._pii_detector.detect)
            | RunnableLambda(self._moderation.detect)
        )

    def invoke(self, ticket):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="security_pipeline"
        ):
            return self._chain.invoke(
                ticket,
                config={
                    "callbacks": [langfuse_handler]
                }
            )

    def batch(self, tickets: list,) -> list[SecureTicket]:
        with langfuse.start_as_current_observation(
            as_type="span",
            name="response_analysis_batch",
        ):
            return self._chain.batch(
                tickets,
                config={
                    "callbacks": [langfuse_handler]
                }
            )
