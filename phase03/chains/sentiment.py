from langchain_core.runnables import RunnableLambda, Runnable
from prompts.sentiment import SENTIMENT_PROMPT
from models.analysis_report import Sentiment
from llms.factory import LLMFactory
from langfuse import get_client
from tracing.client import langfuse_handler

langfuse = get_client()


def build() -> Runnable:
    return (
        RunnableLambda(lambda ticket: ticket.sanitized_text)
        | SENTIMENT_PROMPT
        | LLMFactory.structured(Sentiment)
    )
