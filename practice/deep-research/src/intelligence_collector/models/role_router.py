from __future__ import annotations

from collections.abc import Mapping


DEEP_REASONING_ROLES = frozenset({"planner", "verifier"})
STANDARD_ROLES = frozenset({"supervisor", "worker", "synthesizer", "fixer"})


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
