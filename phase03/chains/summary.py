from langchain_core.runnables import Runnable, RunnableLambda
from models.analysis_report import Summary
from prompts.summary import SUMMARY_PROMPT
from llms.factory import LLMFactory
from langfuse import get_client
from tracing.client import langfuse_handler

langfuse = get_client()


def build() -> Runnable:
    return (
        RunnableLambda(lambda ticket: ticket.sanitized_text)
        | SUMMARY_PROMPT
        | LLMFactory.structured(Summary)
    )
