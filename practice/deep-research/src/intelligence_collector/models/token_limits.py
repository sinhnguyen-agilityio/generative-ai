"""Per-call provider output caps. Never truncate evidence to meet a token budget."""

import json
import os


DEFAULT_OUTPUT_LIMITS = {
    "guardrail": 128,
    "scope": 1536,
    "planner": 1024,
    "researcher": 512,
    "research_compressor": 1536,
    "synthesizer": 2048,
    "fixer": 768,
    "verify": 3072,
    "research_verify": 2048,
    "feedback_router": 256,
    "extension_judge": 512,
}


def output_limits_from_env() -> dict[str, int]:
    if os.getenv("INTELLIGENCE_DISABLE_OUTPUT_CAPS", "").lower() in {"1", "true"}:
        return {}
    overrides = json.loads(os.getenv("INTELLIGENCE_NODE_OUTPUT_TOKENS", "{}"))
    if not isinstance(overrides, dict):
        raise ValueError("INTELLIGENCE_NODE_OUTPUT_TOKENS must be a JSON object")
    if set(overrides) - DEFAULT_OUTPUT_LIMITS.keys():
        raise ValueError("INTELLIGENCE_NODE_OUTPUT_TOKENS contains unknown roles")
    if any(type(value) is not int or not 64 <= value <= 32768
           for value in overrides.values()):
        raise ValueError("Output token limits must be integers between 64 and 32768")
    return {**DEFAULT_OUTPUT_LIMITS, **overrides}


def capped_model(model, role, limits):
    """Clone supported LangChain provider models with the cap before schema/tool binding.

    Do not wrap duck-typed test fakes; a real provider must support model_copy. Provider
    models read these fields when constructing requests, including structured output.
    """
    if not limits or model is None or not hasattr(model, "model_copy"):
        return model
    fields = type(model).model_fields
    field = "max_output_tokens" if "max_output_tokens" in fields else "max_tokens"
    if field not in fields:
        raise ValueError(f"No supported output-token field for {type(model).__name__}")
    limit = limits.get(role, 1024)
    configured = getattr(model, field, None)
    if isinstance(configured, int) and configured > 0:
        limit = min(limit, configured)
    return model.model_copy(update={field: limit})
