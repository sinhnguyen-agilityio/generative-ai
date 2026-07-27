"""Helpers for attaching Langfuse tracing to LangChain runs."""
from typing import Any
from config import settings


def get_langfuse_handler() -> Any | None:
    """Return a Langfuse callback handler when tracing is configured."""
    if not settings.langfuse_public_key:
        return None

    try:
        from langfuse.langchain import CallbackHandler
    except Exception:
        return None

    return CallbackHandler()
