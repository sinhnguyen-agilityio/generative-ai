from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv


MAX_SEARCH_CALLS = 5
MAX_RECOVERY_ATTEMPTS = 10
MAX_CONCURRENT_RESEARCHERS = 5


def _bounded_int(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}") from exc
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


def load_local_env() -> None:
    """Load an ignored local .env file without replacing explicit process settings."""
    load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    vfs_root: str = ".data/vfs"
    checkpoint_path: str = ".data/checkpoints.sqlite3"
    vector_path: str = ".data/vector-index.json"
    chroma_path: str = ".data/chroma"
    embedding_model: str = "text-embedding-3-small"
    evidence_top_k: int = 6
    output_dir: str = "exports"
    max_search_calls: int = 5
    max_concurrent_researchers: int = 3
    recovery_attempts: int = 3

    @classmethod
    def from_env(cls):
        return cls(
            vfs_root=os.getenv("INTELLIGENCE_VFS_ROOT", cls.vfs_root),
            checkpoint_path=os.getenv("INTELLIGENCE_CHECKPOINT_PATH", cls.checkpoint_path),
            vector_path=os.getenv("INTELLIGENCE_VECTOR_PATH", cls.vector_path),
            chroma_path=os.getenv("INTELLIGENCE_CHROMA_PATH", cls.chroma_path),
            embedding_model=os.getenv("INTELLIGENCE_EMBEDDING_MODEL", cls.embedding_model),
            evidence_top_k=_bounded_int("INTELLIGENCE_EVIDENCE_TOP_K", 6, 20),
            output_dir=os.getenv("INTELLIGENCE_OUTPUT_DIR", cls.output_dir),
            max_search_calls=_bounded_int("INTELLIGENCE_MAX_SEARCH_CALLS", 5, MAX_SEARCH_CALLS),
            max_concurrent_researchers=_bounded_int(
                "INTELLIGENCE_MAX_CONCURRENT_RESEARCHERS", 3, MAX_CONCURRENT_RESEARCHERS
            ),
            recovery_attempts=_bounded_int(
                "INTELLIGENCE_RECOVERY_ATTEMPTS", 3, MAX_RECOVERY_ATTEMPTS
            ),
        )


@dataclass(frozen=True)
class RoleModelSettings:
    """Model profiles for deep-reasoning (scope, planner, verify) and standard (researcher,
    research_compressor, synthesizer, fixer) roles. Each profile may have its own fallback."""

    reasoning_model: str
    standard_model: str
    reasoning_provider: str | None = None
    standard_provider: str | None = None
    reasoning_kwargs: dict[str, Any] | None = field(default=None, repr=False)
    standard_kwargs: dict[str, Any] | None = field(default=None, repr=False)
    reasoning_fallback_model: str | None = None
    reasoning_fallback_provider: str | None = None
    standard_fallback_model: str | None = None
    standard_fallback_provider: str | None = None

    @staticmethod
    def _json_object(name: str, default: str) -> dict[str, Any]:
        value = json.loads(os.getenv(name, default))
        if not isinstance(value, dict):
            raise ValueError(f"{name} must contain a JSON object")
        return value

    @classmethod
    def from_env(cls):
        reasoning_model = os.getenv("INTELLIGENCE_REASONING_MODEL", "").strip()
        standard_model = os.getenv("INTELLIGENCE_STANDARD_MODEL", "").strip()
        if not reasoning_model or not standard_model:
            raise ValueError(
                "INTELLIGENCE_REASONING_MODEL and INTELLIGENCE_STANDARD_MODEL are required"
            )
        return cls(
            reasoning_model=reasoning_model,
            standard_model=standard_model,
            reasoning_provider=os.getenv("INTELLIGENCE_REASONING_PROVIDER") or None,
            standard_provider=os.getenv("INTELLIGENCE_STANDARD_PROVIDER") or None,
            reasoning_kwargs=cls._json_object(
                "INTELLIGENCE_REASONING_KWARGS", '{"reasoning_effort":"xhigh"}'
            ),
            standard_kwargs=cls._json_object("INTELLIGENCE_STANDARD_KWARGS", "{}"),
            reasoning_fallback_model=os.getenv("INTELLIGENCE_REASONING_FALLBACK_MODEL") or None,
            reasoning_fallback_provider=os.getenv("INTELLIGENCE_REASONING_FALLBACK_PROVIDER") or None,
            standard_fallback_model=os.getenv("INTELLIGENCE_STANDARD_FALLBACK_MODEL") or None,
            standard_fallback_provider=os.getenv("INTELLIGENCE_STANDARD_FALLBACK_PROVIDER") or None,
        )


@dataclass(frozen=True)
class LangfuseSettings:
    enabled: bool = False
    public_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)
    host: str = "https://cloud.langfuse.com"
    environment: str = "development"
    release: str = "local"

    @classmethod
    def from_env(cls, *, required: bool = False):
        enabled = os.getenv("INTELLIGENCE_LANGFUSE_ENABLED", "false").lower() in {
            "1", "true", "yes", "on",
        }
        settings = cls(
            enabled=enabled,
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "").strip(),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", "").strip(),
            host=(
                os.getenv("LANGFUSE_BASE_URL")
                or os.getenv("LANGFUSE_HOST")
                or "https://cloud.langfuse.com"
            ).strip().rstrip("/"),
            environment=os.getenv("LANGFUSE_ENVIRONMENT", "development").strip(),
            release=os.getenv("LANGFUSE_RELEASE", "local").strip(),
        )
        if required and not settings.enabled:
            raise RuntimeError("Langfuse tracing must be enabled for live execution")
        if settings.enabled and (not settings.public_key or not settings.secret_key):
            raise RuntimeError("Langfuse tracing credentials are not configured")
        return settings
