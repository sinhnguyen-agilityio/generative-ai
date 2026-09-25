from __future__ import annotations

from collections.abc import Mapping


# "verify" is the LLM acceptance-criteria check (agents/acceptance_verifier.py); the
# source-verification script itself makes no LLM call and so has no role here.
DEEP_REASONING_ROLES = frozenset({"scope", "planner", "verify"})
STANDARD_ROLES = frozenset({"researcher", "research_compressor", "synthesizer", "fixer"})


class RoleModelRouter:
    """Route semantic roles to models with the appropriate cost/quality profile."""

    def __init__(self, deep_reasoning, standard, *, role_overrides: Mapping[str, object] | None = None):
        self.deep_reasoning = deep_reasoning
        self.standard = standard
        self.role_overrides = dict(role_overrides or {})

    def router_for(self, role: str):
        if role in self.role_overrides:
            return self.role_overrides[role]
        if role in DEEP_REASONING_ROLES:
            return self.deep_reasoning
        return self.standard

    def invoke(self, role, prompt):
        return self.router_for(role).invoke(role, prompt)

    def invoke_structured(self, role, prompt, schema, **kwargs):
        return self.router_for(role).invoke_structured(role, prompt, schema, **kwargs)

    def invoke_messages(self, role, messages, schema, **kwargs):
        return self.router_for(role).invoke_messages(role, messages, schema, **kwargs)

    def invoke_with_tools(self, role, messages, tools, **kwargs):
        return self.router_for(role).invoke_with_tools(role, messages, tools, **kwargs)
