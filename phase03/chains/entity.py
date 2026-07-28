from langchain_core.runnables import Runnable, RunnableLambda
from prompts.entity import ENTITY_PROMPT
from models.analysis_report import Entities
from llms.factory import LLMFactory
from langfuse import get_client
from tracing.client import langfuse_handler

langfuse = get_client()


def build() -> Runnable:
    chain = (
        RunnableLambda(lambda ticket: ticket.sanitized_text)
        | ENTITY_PROMPT
        | LLMFactory.structured(Entities)
    )

    def run(input):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="summary"
        ):
            return chain.invoke(
                input,
                config={
                    "callbacks": [langfuse_handler]
                },
            )

    return RunnableLambda(run)
