from langchain_core.runnables import Runnable, RunnableLambda
from prompts.entity import ENTITY_PROMPT
from models.analysis_report import Entities
from llms.factory import LLMFactory
from langfuse import get_client
from tracing.client import langfuse_handler

langfuse = get_client()


def build() -> Runnable:
    return (
        RunnableLambda(lambda ticket: ticket.sanitized_text)
        | ENTITY_PROMPT
        | LLMFactory.structured(Entities)
    )
