from langfuse.langchain import CallbackHandler


def get_langfuse_handler() -> CallbackHandler:
    """
    Create a Langfuse callback handler.

    The handler automatically traces:
    - LLM generations
    - Tool calls
    - Agent execution
    - Token usage
    - Latency
    """

    return CallbackHandler()
