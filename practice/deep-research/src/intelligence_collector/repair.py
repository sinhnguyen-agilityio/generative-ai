"""Repair diagnosis: the smallest useful correction for a failed verification.

Deterministic, no model call. Each failed check maps to one route:

* ``presentation`` — language, length, format or citation layout; edit presentation only.
* ``vfs``          — the stored evidence holds the answer (or a sentence is unsupported); rewrite
                     the affected answers from the full stored evidence.
* ``chroma``       — evidence is missing from what the writer saw but the question's stored
                     corpus is larger; locate passages with the vector index, then read their
                     VFS captures through provenance.
* ``research``     — the stored evidence lacks the answer; targeted Tavily research for the
                     affected task inside the approved plan.
* ``replan``       — targeted research for that item already failed; propose a replacement plan
                     and wait for approval.

The run takes the largest route any issue needs (``replan`` > ``research`` > ``chroma`` > ``vfs``
> ``presentation``); the writer still receives every issue, so one cycle fixes all of them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from intelligence_collector.contracts import ResearchTodo, VerificationResult
from intelligence_collector.source_verification import EvidenceBundle, SourceCatalog, _normalize_url

ROUTE_ORDER = ("replan", "research", "chroma", "vfs", "presentation")
_WORD = re.compile(r"[a-z0-9]+")


def item_key(text: str) -> str:
    return " ".join(_WORD.findall(text.casefold()))


def _words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.casefold()) if len(word) > 2}


def affected_todo(item: str, missing: list[str], todos: list[ResearchTodo]) -> ResearchTodo | None:
    """The research task whose question best overlaps the unanswered item."""
    if not todos:
        return None
    wanted = _words(" ".join([item, *missing]))
    return max(todos, key=lambda todo: len(wanted & _words(todo.research_question)))


def diagnose(
    verification: VerificationResult,
    *,
    todos: list[ResearchTodo],
    full_catalog: SourceCatalog,
    shown_source_ids: dict[str, list[str]],
    researched_items: list[str],
    can_replan: bool,
) -> dict:
    """Repair plan for the failed checks of `verification`.

    `shown_source_ids` maps todo id -> source ids the writer saw for the draft;
    `researched_items` are item keys that already received targeted research.
    """
    bundles = {bundle.todo_id: bundle for bundle in full_catalog.bundles}
    issues, passing = [], []
    for check in verification.check_results:
        if check.passed:
            if (
                check.gate == "coverage"
                and check.item
                and not check.item.startswith(("Output requirement", "Satisfies this reviewer"))
            ):
                passing.append(check.item)
            continue
        todo = None
        if check.gate == "citation":
            route = "vfs" if check.issue_type == "draft_defect" else "presentation"
        elif check.gate == "faithfulness":
            route = "vfs"
        elif check.issue_type == "presentation":
            route = "presentation"
        elif check.issue_type == "draft_defect":
            route = "vfs"
            todo = affected_todo(check.item, check.missing_points, todos)
        else:  # coverage evidence gap
            todo = affected_todo(check.item, check.missing_points, todos)
            bundle = bundles.get(todo.todo_id) if todo else None
            unseen = bundle is not None and set(bundle.source_ids) - set(
                shown_source_ids.get(todo.todo_id, bundle.source_ids)
            )
            if unseen:
                route = "chroma"
            elif item_key(check.item) in researched_items:
                route = "replan" if can_replan else "vfs"
            elif todo is not None:
                route = "research"
            else:
                route = "vfs"
        issues.append(
            {
                "check": check.name,
                "gate": check.gate,
                "issue_type": check.issue_type,
                "item": check.item,
                "missing_points": list(check.missing_points),
                "detail": "; ".join(check.details),
                "route": route,
                "todo_id": todo.todo_id if todo else "",
            }
        )
    route = next((r for r in ROUTE_ORDER if any(i["route"] == r for i in issues)), "vfs")
    return {"route": route, "issues": issues, "passing": passing}


def locate_with_index(
    vfs,
    index,
    run_id: str,
    catalog: SourceCatalog,
    writer_catalog: SourceCatalog,
    issues: list[dict],
    artifacts: list[dict],
    *,
    limit: int = 4,
) -> SourceCatalog:
    """Chroma route: retrieve passages for each affected item, read each hit's original capture
    from the VFS through its artifact provenance, and add those sources to the writer's view of
    the affected task. Hits that cannot be traced back to a VFS capture are ignored."""
    paths = {
        item.get("artifact_id"): item.get("path") for item in artifacts if item.get("artifact_id")
    }
    by_url = {_normalize_url(source.url): sid for sid, source in catalog.sources.items()}
    added: dict[str, list[str]] = {}
    for issue in issues:
        if issue["route"] != "chroma" or not issue["todo_id"]:
            continue
        query = " ".join([issue["item"], *issue["missing_points"]])
        for chunk in index.retrieve(query, limit=limit, run_id=run_id, todo_id=issue["todo_id"]):
            path = paths.get(chunk.artifact_id)
            if not path:
                continue
            base = (vfs.root / run_id).resolve()
            capture = json.loads(vfs.read(run_id, str(Path(path).resolve().relative_to(base))))
            sid = by_url.get(_normalize_url(capture["source"]["locator"]))
            if sid:
                added.setdefault(issue["todo_id"], []).append(sid)
    bundles = []
    for bundle in writer_catalog.bundles:
        extra = [sid for sid in added.get(bundle.todo_id, []) if sid not in bundle.source_ids]
        bundles.append(
            EvidenceBundle(
                bundle.todo_id,
                bundle.question,
                bundle.summary,
                [*bundle.source_ids, *extra],
                bundle.gaps,
            )
        )
    return SourceCatalog(sources=catalog.sources, bundles=bundles)
