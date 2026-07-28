from langchain_core.runnables import RunnableLambda
from prompts.urgency import URGENCY_PROMPT
from models.analysis_report import Urgency
from llms.factory import LLMFactory
from langfuse import get_client
from tracing.client import langfuse_handler

langfuse = get_client()


def build():
    chain = (RunnableLambda(
        lambda report: {
            "report": report.model_dump_json(indent=2)
        })
        | URGENCY_PROMPT
        | LLMFactory.structured(Urgency))

    def run(input):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="ugency"
        ):
            return chain.invoke(
                input,
                config={
                    "callbacks": [langfuse_handler]
                },
            )

    return RunnableLambda(run)
