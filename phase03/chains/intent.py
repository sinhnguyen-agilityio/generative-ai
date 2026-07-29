# workflows/primitives/intent_chain.py
from langchain_core.runnables import Runnable, RunnableLambda
from prompts.intent import INTENT_PROMPT
from models.analysis_report import Intent
from llms.factory import LLMFactory
from langfuse import get_client
from tracing.client import langfuse_handler

langfuse = get_client()


def build() -> Runnable:
    return (
        RunnableLambda(lambda ticket: ticket.sanitized_text)
        | INTENT_PROMPT
        | LLMFactory.structured(Intent)
    )
