from langchain_core.runnables import RunnableLambda
from prompts.escalation import ESCALATION_PROMPT
from models.analysis_report import Escalation
from llms.factory import LLMFactory
from langfuse import get_client
from tracing.client import langfuse_handler

langfuse = get_client()


def build():
    chain = (
        RunnableLambda(
            lambda report: {
                "report": report.model_dump_json(indent=2)
            }
        )
        | ESCALATION_PROMPT
        | LLMFactory.structured(Escalation)
    )

    def run(input):
            with langfuse.start_as_current_observation(
                as_type="span",
                name="escalation"
            ):
                return chain.invoke(
                    input,
                    config={
                        "callbacks": [langfuse_handler]
                    },
                )

    return RunnableLambda(run)