"""One model tier: a primary chat model, an optional fallback, and validated structured output."""

import json
from typing import TypeVar

from pydantic import BaseModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable, RunnableLambda

from intelligence_collector.models.redaction import redact_text, redact_value
from intelligence_collector.models.token_limits import DEFAULT_OUTPUT_LIMITS, capped_model

T = TypeVar("T", bound=BaseModel)
MAX_MODEL_ATTEMPTS = 3
# HTTP statuses that mean "the provider is slow/overloaded", not "the model answered badly".
TRANSIENT_STATUS = frozenset({408, 429, 500, 502, 503, 504})


def _is_transient(exc: BaseException) -> bool:
    """True for timeouts, dropped connections, rate limits and provider overloads.

    Calling again immediately only repeats the wait (a timeout costs the full timeout again), and
    the fallback model already had its turn inside the chain, so these are not retried. A model
    that answered badly (invalid JSON, schema violation, a provider 400 such as Groq's flaky
    `json_validate_failed`) is different: another attempt often succeeds, so it is retried.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    if getattr(exc, "status_code", None) in TRANSIENT_STATUS or getattr(exc, "code", None) in TRANSIENT_STATUS:
        return True
    name = type(exc).__name__.lower()  # e.g. openai.APITimeoutError, httpx.ReadTimeout
    return "timeout" in name or "connection" in name


def _model_name(model) -> str:
    """Best-effort display name for traces (`model_name`, then `model`, then the class name)."""
    return str(getattr(model, "model_name", None) or getattr(model, "model", None) or type(model).__name__)


def _as_runnable(model, role: str, config) -> Runnable:
    """Make `model` usable with `with_fallbacks`, which only accepts Runnables.

    Real chat models already are Runnables. Duck-typed models (test fakes exposing `.invoke()`,
    or plain `(role, prompt)` callables) are wrapped. `config` is forwarded to a fake's
    `.invoke()` only when callbacks are configured, as fakes may not accept the argument.
    """
    if isinstance(model, Runnable):
        return model
    if hasattr(model, "invoke"):
        return RunnableLambda(lambda prompt: model.invoke(prompt, config=config) if config else model.invoke(prompt))
    if callable(model):
        return RunnableLambda(lambda prompt: model(role, prompt))
    raise TypeError("configured model must be callable or expose invoke()")


def _bind_schema(model, schema):
    """Return `model` constrained to emit `schema` via the provider's native structured output
    (Gemini response schema, Groq json_schema) instead of trusting free-text JSON, which Gemini
    often wraps in Markdown fences. Models without `with_structured_output` are returned as-is
    and their raw output is parsed by `ModelRouter._parse`."""
    if not hasattr(model, "with_structured_output"):
        return model
    if type(model).__name__ == "ChatGroq":
        # ChatGroq defaults to function calling, and gpt-oss models there sometimes call a tool
        # literally named "json" instead of the bound schema tool, which Groq rejects server-side
        # (400 tool_use_failed); its response_format path has no tools to misname. strict=True
        # turns on constrained decoding for gpt-oss-20b/120b, which *guarantees* schema-valid JSON
        # instead of best-effort compliance -- without it a complex schema can yield malformed or
        # empty output that Groq rejects post-hoc. Groq ignores strict for models that don't
        # support it, so it is safe to pass unconditionally.
        return model.with_structured_output(schema, method="json_schema", strict=True)
    return model.with_structured_output(schema)


def _bind_tools(model, tools: list, tool_choice: str | None):
    """Bind `tools` to `model`; models that cannot bind tools are returned as-is."""
    if not hasattr(model, "bind_tools"):
        return model
    return model.bind_tools(tools, tool_choice=tool_choice) if tool_choice else model.bind_tools(tools)


def _render_prompt(prompt) -> str:
    """Render the exact model input as readable, redacted text for trace inspection."""
    if not isinstance(prompt, list):
        return redact_text(prompt)
    blocks = []
    for message in prompt:
        kind = getattr(message, "type", type(message).__name__).upper()
        blocks.append(f"[{kind}]\n{redact_text(getattr(message, 'content', message))}")
    return "\n\n".join(blocks)


class ModelRouter:
    """One model tier: a primary model with an optional fallback (LangChain `with_fallbacks`).

    Two independent recovery layers:
    * provider errors (timeouts, rate limits) fall through to the fallback model, once per call;
    * invalid structured output (bad JSON / schema violation) re-invokes the whole primary ->
      fallback chain, up to `max_attempts` times, before raising `ValueError`. Transient provider
      failures (see `_is_transient`) are raised at once instead of being retried.
    """

    def __init__(self, primary, fallback=None, *, max_attempts: int = 2, trace=None, callbacks=None,
                 output_limits=None):
        if primary is None:
            raise RuntimeError("model router requires a configured primary model")
        if not isinstance(max_attempts, int) or not 1 <= max_attempts <= MAX_MODEL_ATTEMPTS:
            raise ValueError(f"max_attempts must be between 1 and {MAX_MODEL_ATTEMPTS}")
        self.primary, self.fallback = primary, fallback
        self.max_attempts = max_attempts
        self.trace = trace  # called as trace(**event); None disables tracing (and its rendering cost)
        self.callbacks = list(callbacks or [])
        self.output_limits = dict(DEFAULT_OUTPUT_LIMITS if output_limits is None else output_limits)
        # A fallback that rescues a call is visible in the callback trace as the child model run.
        self.model_name = _model_name(primary)

    # ----- public API -----

    def invoke_messages(
        self, role: str, messages: list[BaseMessage], schema: type[T], *,
        run_id: str = "", thread_id: str = "",
    ) -> T:
        """Call the model and return its output validated as `schema`.

        Retries up to `max_attempts` when the output is invalid; every attempt is traced.
        """
        config = self._config(role, run_id, thread_id)
        chain = self._chain(
            _bind_schema(capped_model(self.primary, role, self.output_limits), schema),
            _bind_schema(capped_model(self.fallback, role, self.output_limits), schema), role, config)
        errors = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                result = self._parse(role, chain.invoke(messages, config=config), schema)
            except Exception as exc:
                transient = _is_transient(exc)
                errors.append(redact_text(exc))
                # The raw error text is withheld from the trace: it can echo model output.
                message = (
                    f"{type(exc).__name__}: model call failed"
                    if transient else f"{type(exc).__name__}: invalid structured output"
                )
                self._emit(
                    role, run_id, thread_id, attempt, messages,
                    output_text=message, validation_error=message,
                )
                if transient:
                    raise RuntimeError(f"model call for {role} failed: {errors[-1]}") from exc
                continue
            self._emit(
                role, run_id, thread_id, attempt, messages,
                output_text=redact_text(result.model_dump_json(indent=2)),
                decision=redact_value(result.model_dump(mode="json")),
                message_count=len(messages),
            )
            return result
        raise ValueError(f"invalid structured output for {role} after {self.max_attempts} attempts: {'; '.join(errors)}")

    def invoke_with_tools(
        self, role: str, messages: list[BaseMessage], tools: list, *,
        run_id: str = "", thread_id: str = "", tool_choice: str | None = None,
    ) -> AIMessage:
        """Tool-calling decision: the returned AIMessage carries a final answer or tool calls."""
        config = self._config(role, run_id, thread_id)
        chain = self._chain(
            _bind_tools(capped_model(self.primary, role, self.output_limits), tools, tool_choice),
            None if self.fallback is None else _bind_tools(
                capped_model(self.fallback, role, self.output_limits), tools, tool_choice),
            role, config,
        )
        raw = chain.invoke(messages, config=config)
        if not isinstance(raw, AIMessage):
            raise TypeError(f"tool-calling model output for {role} must be an AIMessage")
        decision = redact_value({
            "content": raw.content,
            "tool_calls": [{"name": call["name"], "args": call["args"]} for call in raw.tool_calls],
        })
        self._emit(
            role, run_id, thread_id, 1, messages,
            output_text=json.dumps(decision, ensure_ascii=False, indent=2, default=str),
            decision=decision,
            message_count=len(messages),
        )
        return raw

    # ----- internals -----

    def _config(self, role: str, run_id: str, thread_id: str) -> dict | None:
        """LangChain run config so callbacks (Langfuse) group model runs by session and role.
        None when no callbacks are configured."""
        if not self.callbacks:
            return None
        return {
            "callbacks": self.callbacks,
            "run_name": f"model.{role}",
            "metadata": {
                "langfuse_session_id": thread_id,
                "run_id": run_id,
                "thread_id": thread_id,
                "role": role,
            },
        }

    @staticmethod
    def _chain(primary, fallback, role: str, config) -> Runnable:
        """primary, then `fallback` if the primary raises. Built once per call, not per attempt."""
        chain = _as_runnable(primary, role, config)
        if fallback is not None:
            chain = chain.with_fallbacks([_as_runnable(fallback, role, config)])
        return chain

    @staticmethod
    def _parse(role: str, raw, schema: type[T]) -> T:
        """Coerce whatever the model returned (schema instance, dict, message or JSON text)
        into a validated `schema` instance."""
        if isinstance(raw, BaseMessage):
            # `.text` collapses provider-specific multi-block content (e.g. Gemini's thinking +
            # text blocks under reasoning_effort) to the final text block; `.content` can be a
            # list of blocks rather than a string.
            raw = raw.text
        elif hasattr(raw, "content"):
            raw = raw.content
        if isinstance(raw, schema):
            return raw
        if isinstance(raw, dict):
            return schema.model_validate(raw)
        if isinstance(raw, str):
            return schema.model_validate(json.loads(raw))
        raise TypeError(f"model output for {role} is not structured")

    def _emit(self, role, run_id, thread_id, attempt, messages, **fields) -> None:
        """Send one trace event. Rendering the prompt is skipped when nobody is listening."""
        if not self.trace:
            return
        self.trace(
            role=role,
            run_id=run_id,
            thread_id=thread_id,
            selected_model=self.model_name,
            attempt=attempt,
            input_text=_render_prompt(messages),
            **fields,
        )
