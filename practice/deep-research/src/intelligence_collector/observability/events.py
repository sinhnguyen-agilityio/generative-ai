from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WorkflowEvent:
    name: str
    run_id: str
    stage: str
    task_id: str | None = None
    retryable: bool = False
    artifact_ids: tuple[str, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowError:
    message: str
    run_id: str
    stage: str
    task_id: str | None = None
    retryable: bool = False
    can_resume: bool = False
    needs_review: bool = False
    artifact_ids: tuple[str, ...] = ()
