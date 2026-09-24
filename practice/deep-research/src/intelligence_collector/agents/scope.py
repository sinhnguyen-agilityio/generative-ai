from __future__ import annotations
import hashlib
from intelligence_collector.contracts import ScopedRequest
from intelligence_collector.storage.vfs import VFS

REQUIRED = ("objective", "research_boundaries", "constraints", "success_criteria", "deliverable_format")
def missing_scope_fields(scope: dict[str, str]) -> list[str]:
    return [key for key in REQUIRED if not scope.get(key, "").strip()]

def narrowing_suggestions(scope: dict[str, str]) -> list[str]:
    """Return actionable narrowing prompts for broad or underspecified requests."""
    suggestions = []
    if not scope.get("research_boundaries", "").strip():
        suggestions.append("Limit the topic to a defined audience, time period, or technical scope.")
    if not scope.get("success_criteria", "").strip():
        suggestions.append("State what evidence would make the research answer successful.")
    if not scope.get("deliverable_format", "").strip():
        suggestions.append("Choose one deliverable format, such as a comparison or technical report.")
    if len(scope.get("objective", "")) > 1000:
        suggestions.append("Split the objective into one primary question and explicit exclusions.")
    return suggestions

def scope_ready(scope: dict[str, str]) -> bool:
    return all(scope.get(k, "").strip() for k in REQUIRED)
def build_scope(scope: dict[str, str], vfs: VFS | None = None, run_id: str | None = None) -> ScopedRequest:
    if not scope_ready(scope):
        raise ValueError("scope is incomplete: " + ", ".join(missing_scope_fields(scope)))
    value = "|".join(scope[k].strip() for k in REQUIRED)
    result = ScopedRequest(**scope, scope_hash=hashlib.sha256(value.encode()).hexdigest())
    if vfs and run_id:
        vfs.write_json(run_id, "scope/brief_summary.v1.json", result.model_dump(mode="json"), "brief_summary")
    return result
