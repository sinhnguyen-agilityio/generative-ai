from langchain_core.runnables import Runnable, RunnableLambda

from middlewares.toxicity import OpenAIModerationDetector
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

    def build(self) -> Runnable:
        chain = (
            RunnableLambda(self._pii_detector.detect)
            | RunnableLambda(self._moderation.detect)
        )

        def invoke(ticket):
            with langfuse.start_as_current_observation(
                as_type="span",
                name="security_pipeline"
            ):
                return chain.invoke(
                    ticket,
                    config={
                        "callbacks": [langfuse_handler]
                    }
                )

        return RunnableLambda(invoke)
