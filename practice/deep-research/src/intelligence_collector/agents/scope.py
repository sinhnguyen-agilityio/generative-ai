from __future__ import annotations
import hashlib
from intelligence_collector.contracts import ScopedRequest

REQUIRED = ("objective", "research_boundaries", "constraints", "success_criteria", "deliverable_format")
def scope_ready(scope: dict[str, str]) -> bool:
    return all(scope.get(k, "").strip() for k in REQUIRED)
def build_scope(scope: dict[str, str]) -> ScopedRequest:
    if not scope_ready(scope): raise ValueError("scope is incomplete")
    value = "|".join(scope[k].strip() for k in REQUIRED)
    return ScopedRequest(**scope, scope_hash=hashlib.sha256(value.encode()).hexdigest())

