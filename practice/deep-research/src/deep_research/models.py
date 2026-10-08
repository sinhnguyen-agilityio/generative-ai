"""Chat models. Reasoning tier: clarify, brief, planner, supervisor, verifier, triage.
Standard tier: guardrail, researcher, topic report, report writer."""

from __future__ import annotations

from langchain.chat_models import init_chat_model

from deep_research.config import ModelSpec


def _create(model: str, provider: str | None, timeout: float, kwargs: dict):
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
    if llm._llm_type == "groq-chat":
        return llm.with_structured_output(schema, method="json_schema", strict=True)
    return llm.with_structured_output(schema)


class Tier:
    """A primary chat model plus an optional fallback, tried in that order."""

    def __init__(self, spec: ModelSpec):
        self.primary = _create(spec.model, spec.provider, spec.timeout, spec.kwargs)
        self.fallback = (
            _create(spec.fallback_model, spec.fallback_provider, spec.timeout, {})
            if spec.fallback_model
            else None
        )

    def _wrap(self, build):
        runnable = build(self.primary)
        return runnable.with_fallbacks([build(self.fallback)]) if self.fallback else runnable

    def structured(self, schema):
        return self._wrap(lambda llm: _structured(llm, schema))

    def tools(self, tools):
        return self._wrap(lambda llm: llm.bind_tools(tools))

    def plain(self):
        return self._wrap(lambda llm: llm)
