"""Configure provider-compatible chat models and structured output."""

from __future__ import annotations

from langchain.chat_models import init_chat_model


def _create(model: str, provider: str | None, timeout: float, kwargs: dict):
    """Initialize a chat model with provider options and compatible reasoning/tool support.

    Args:
        model: Chat model identifier.
        provider: Optional LangChain provider name.
        timeout: Request timeout in seconds.
        kwargs: Additional provider-specific model options.

    Returns:
        Configured chat model; reasoning-enabled OpenAI models may use the Responses API.
    """
    args = {"timeout": timeout, **kwargs}
    if provider:
        args["model_provider"] = provider
    llm = init_chat_model(model, **args)
    # OpenAI rejects tools + reasoning_effort on chat completions; the Responses API accepts both.
    if getattr(llm, "use_responses_api", False) is None and getattr(llm, "reasoning_effort", None):
        llm = init_chat_model(model, **args, use_responses_api=True)
    return llm


def _structured(llm, schema):
    # gpt-oss on Groq may rename the tool it must call ("safetyVerdict"), and Groq rejects the
    # call; Groq's structured outputs constrain the JSON to the schema instead (strict for gpt-oss).
    """Configure schema-constrained output, using strict JSON schema for Groq.

    Args:
        llm: Chat model to configure.
        schema: Output schema accepted by the model integration.

    Returns:
        Runnable producing structured results.
    """
    if llm._llm_type == "groq-chat":
        return llm.with_structured_output(schema, method="json_schema", strict=True)
    return llm.with_structured_output(schema)
