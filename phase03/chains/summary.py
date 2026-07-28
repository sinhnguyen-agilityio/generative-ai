from langchain_core.runnables import Runnable, RunnableLambda
from models.analysis_report import Summary
from prompts.summary import SUMMARY_PROMPT
from llms.factory import LLMFactory
from langfuse import get_client
from tracing.client import langfuse_handler

langfuse = get_client()


def build() -> Runnable:

    chain = (
        RunnableLambda(lambda ticket: ticket.sanitized_text)
        | SUMMARY_PROMPT
        | LLMFactory.structured(Summary)
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
