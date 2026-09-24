from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv


def load_local_env() -> None:
    """Load an ignored local .env file without replacing explicit process settings."""
    load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    vfs_root: str = ".data/vfs"
    checkpoint_path: str = ".data/checkpoints.json"
    vector_path: str = ".data/vector-index.json"
    output_dir: str = "exports"
    max_search_calls: int = 5
    recovery_attempts: int = 3

    @classmethod
    def from_env(cls):
        return cls(
            vfs_root=os.getenv("INTELLIGENCE_VFS_ROOT", cls.vfs_root),
            checkpoint_path=os.getenv("INTELLIGENCE_CHECKPOINT_PATH", cls.checkpoint_path),
            vector_path=os.getenv("INTELLIGENCE_VECTOR_PATH", cls.vector_path),
            output_dir=os.getenv("INTELLIGENCE_OUTPUT_DIR", cls.output_dir),
            max_search_calls=min(int(os.getenv("INTELLIGENCE_MAX_SEARCH_CALLS", "5")), 5),
            recovery_attempts=max(1, int(os.getenv("INTELLIGENCE_RECOVERY_ATTEMPTS", "3"))),
        )


@dataclass(frozen=True)
class RoleModelSettings:
    """Model profiles for deep-reasoning and routine implementation roles."""

    reasoning_model: str
    standard_model: str
    reasoning_provider: str | None = None
    standard_provider: str | None = None
    reasoning_kwargs: dict[str, Any] | None = None
    standard_kwargs: dict[str, Any] | None = None

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
        )


@dataclass(frozen=True)
class LangfuseSettings:
    enabled: bool = False
    public_key: str = ""
    secret_key: str = ""
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
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            environment=os.getenv("LANGFUSE_ENVIRONMENT", "development"),
            release=os.getenv("LANGFUSE_RELEASE", "local"),
        )
        if required and not settings.enabled:
            raise RuntimeError("Langfuse tracing must be enabled for live execution")
        if settings.enabled and (not settings.public_key or not settings.secret_key):
            raise RuntimeError("Langfuse tracing credentials are not configured")
        return settings
