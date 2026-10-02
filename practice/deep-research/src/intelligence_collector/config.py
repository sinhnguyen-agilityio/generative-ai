"""Settings read from the environment: storage paths and research limits, the two model tiers, and
Langfuse.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


MAX_SEARCH_CALLS = 5
MAX_SOURCES_PER_TASK = 10
MAX_RECOVERY_ATTEMPTS = 10
MAX_CONCURRENT_RESEARCHERS = 5

# Report Judge guideline checklist (see agents/report_judge.py) -- fixed formatting/language
# rules every report must clear, alongside the judge's fact check. An unmet guideline is
# fixed like any other failure: the Build-Verify-Fix loop reflects and rewrites the report.
DEFAULT_REPORT_GUIDELINES: list[str] = [
    "The report MUST be written in Markdown format.",
    # Matches the writer's own layout (agents/prompts.py::REPORT_WRITER_PROMPT): sources are cited
    # inline as [S1] markers that resolve to hyperlinks in the "## Sources" list, and the "## Scope"
    # section makes no claims. Do not ask for inline URLs here: the writer is told never to put
    # URLs outside the Sources list, so such a rule can never be satisfied.
    "Every body section (each '## ' section except '## Scope', '## Limitations' and "
    "'## Sources') MUST support its claims with inline source citations such as [S1] that match "
    "entries in the '## Sources' list, which gives each source's title and hyperlink. If no "
    "source supports a section, remove it or merge it into the previous section.",
    "The report MUST be written in English.",
]


def _bounded_int(name: str, default: int, maximum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer between 1 and {maximum}") from exc
    if not 1 <= value <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}")
    return value


EVAL_METRICS = ("faithfulness", "context_precision", "response_relevancy")


def _eval_metrics() -> tuple[str, ...]:
    """``INTELLIGENCE_EVAL_METRICS``: comma-separated subset of ``EVAL_METRICS`` (default: all;
    empty: none). Raises ``ValueError`` for an unknown metric name."""
    raw = os.getenv("INTELLIGENCE_EVAL_METRICS")
    if raw is None:
        return EVAL_METRICS
    names = tuple(dict.fromkeys(n.strip().lower() for n in raw.split(",") if n.strip()))
    unknown = [n for n in names if n not in EVAL_METRICS]
    if unknown:
        raise ValueError(
            f"INTELLIGENCE_EVAL_METRICS has unknown metric(s) {unknown}; use {list(EVAL_METRICS)}"
        )
    return names


def _json_string_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)
    value = json.loads(raw)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{name} must be a JSON array of strings")
    return value


def load_local_env() -> None:
    """Load an ignored local .env file without replacing explicit process settings."""
    load_dotenv(override=False)


@dataclass(frozen=True)
class Settings:
    """Storage paths, research limits, report guidelines and the RAGAS judge, read from
    ``INTELLIGENCE_*`` environment variables.
    """

    vfs_root: str = ".data/vfs"
    checkpoint_path: str = ".data/checkpoints.sqlite3"
    run_registry_path: str = ".data/run_registry.sqlite3"
    vector_path: str = ".data/vector-index.json"
    chroma_path: str = ".data/chroma"
    embedding_model: str = "gemini-embedding-2"
    embedding_provider: str = "google_genai"
    evidence_top_k: int = 6
    output_dir: str = "exports"
    max_search_calls: int = 5
    # Distinct pages one research sub-agent keeps for its todo, across all of its searches.
    max_sources_per_task: int = 5
    max_concurrent_researchers: int = 3
    recovery_attempts: int = 2
    faithfulness_model: str = "gpt-4.1-mini"
    faithfulness_provider: str = "openai"
    # RAGAS metrics scored on the final report (judge = the faithfulness model above).
    eval_metrics: tuple[str, ...] = EVAL_METRICS
    report_guidelines: list[str] = field(
        default_factory=lambda: list(DEFAULT_REPORT_GUIDELINES))

    @classmethod
    def from_env(cls):
        """Build the settings from the environment. Raises ``ValueError`` for an out-of-range limit
        or malformed JSON.
        """
        checkpoint_path = os.getenv(
            "INTELLIGENCE_CHECKPOINT_PATH", cls.checkpoint_path
        )
        registry_default = str(
            Path(checkpoint_path).expanduser().with_name("run_registry.sqlite3")
        )
        return cls(
            vfs_root=os.getenv("INTELLIGENCE_VFS_ROOT", cls.vfs_root),
            checkpoint_path=checkpoint_path,
            run_registry_path=os.getenv(
                "INTELLIGENCE_RUN_REGISTRY_PATH", registry_default),
            vector_path=os.getenv("INTELLIGENCE_VECTOR_PATH", cls.vector_path),
            chroma_path=os.getenv("INTELLIGENCE_CHROMA_PATH", cls.chroma_path),
            embedding_model=os.getenv(
                "INTELLIGENCE_EMBEDDING_MODEL", cls.embedding_model),
            embedding_provider=os.getenv(
                "INTELLIGENCE_EMBEDDING_PROVIDER", cls.embedding_provider
            ),
            evidence_top_k=_bounded_int("INTELLIGENCE_EVIDENCE_TOP_K", 6, 20),
            output_dir=os.getenv("INTELLIGENCE_OUTPUT_DIR", cls.output_dir),
            max_search_calls=_bounded_int(
                "INTELLIGENCE_MAX_SEARCH_CALLS", 5, MAX_SEARCH_CALLS),
            max_sources_per_task=_bounded_int(
                "INTELLIGENCE_MAX_SOURCES_PER_TASK", 5, MAX_SOURCES_PER_TASK
            ),
            max_concurrent_researchers=_bounded_int(
                "INTELLIGENCE_MAX_CONCURRENT_RESEARCHERS", 3, MAX_CONCURRENT_RESEARCHERS
            ),
            recovery_attempts=_bounded_int(
                "INTELLIGENCE_RECOVERY_ATTEMPTS", 3, MAX_RECOVERY_ATTEMPTS
            ),
            report_guidelines=_json_string_list(
                "INTELLIGENCE_REPORT_GUIDELINES", DEFAULT_REPORT_GUIDELINES
            ),
            faithfulness_model=os.getenv(
                "INTELLIGENCE_FAITHFULNESS_MODEL", cls.faithfulness_model
            ),
            faithfulness_provider=os.getenv(
                "INTELLIGENCE_FAITHFULNESS_PROVIDER", cls.faithfulness_provider
            ),
            eval_metrics=_eval_metrics(),
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
    # Per-request timeout (seconds) for each tier's models, fallback included. The reasoning tier
    # gets more time: a reasoning model (e.g. gpt-5-nano at reasoning_effort=high) can spend well
    # over 30s thinking before it emits the first token.
    reasoning_timeout: float = 120.0
    standard_timeout: float = 60.0

    @staticmethod
    def _bounded_seconds(name: str, default: float) -> float:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(
                f"{name} must be a number of seconds between 5 and 600") from exc
        if not 5 <= value <= 600:
            raise ValueError(f"{name} must be between 5 and 600 seconds")
        return value

    @staticmethod
    def _json_object(name: str, default: str) -> dict[str, Any]:
        value = json.loads(os.getenv(name, default))
        if not isinstance(value, dict):
            raise ValueError(f"{name} must contain a JSON object")
        return value

    @classmethod
    def from_env(cls):
        """Read both tiers from ``INTELLIGENCE_REASONING_*`` and ``INTELLIGENCE_STANDARD_*``; both
        model names are required.
        """
        reasoning_model = os.getenv("INTELLIGENCE_REASONING_MODEL", "").strip()
        standard_model = os.getenv("INTELLIGENCE_STANDARD_MODEL", "").strip()
        if not reasoning_model or not standard_model:
            raise ValueError(
                "INTELLIGENCE_REASONING_MODEL and INTELLIGENCE_STANDARD_MODEL are required"
            )
        return cls(
            reasoning_model=reasoning_model,
            standard_model=standard_model,
            reasoning_provider=os.getenv(
                "INTELLIGENCE_REASONING_PROVIDER") or None,
            standard_provider=os.getenv(
                "INTELLIGENCE_STANDARD_PROVIDER") or None,
            # max_tokens guards against reasoning_effort="high" silently exhausting its whole
            # budget on the hidden reasoning trace and emitting zero characters of the actual
            # JSON answer (surfaces as an empty failed_generation / json_validate_failed on Groq)
            # -- a larger structured-output schema (e.g. more judge criteria) makes that trace
            # longer, so a low provider default becomes more likely to be hit.
            reasoning_kwargs=cls._json_object(
                "INTELLIGENCE_REASONING_KWARGS", '{"reasoning_effort":"high","max_tokens":8000}'
            ),
            standard_kwargs=cls._json_object(
                "INTELLIGENCE_STANDARD_KWARGS", "{}"),
            reasoning_fallback_model=os.getenv(
                "INTELLIGENCE_REASONING_FALLBACK_MODEL") or None,
            reasoning_fallback_provider=os.getenv(
                "INTELLIGENCE_REASONING_FALLBACK_PROVIDER") or None,
            standard_fallback_model=os.getenv(
                "INTELLIGENCE_STANDARD_FALLBACK_MODEL") or None,
            standard_fallback_provider=os.getenv(
                "INTELLIGENCE_STANDARD_FALLBACK_PROVIDER") or None,
            reasoning_timeout=cls._bounded_seconds(
                "INTELLIGENCE_REASONING_TIMEOUT", cls.reasoning_timeout),
            standard_timeout=cls._bounded_seconds(
                "INTELLIGENCE_STANDARD_TIMEOUT", cls.standard_timeout),
        )


@dataclass(frozen=True)
class LangfuseSettings:
    """Langfuse connection settings; tracing is off unless ``INTELLIGENCE_LANGFUSE_ENABLED`` is
    true.
    """

    enabled: bool = False
    public_key: str = field(default="", repr=False)
    secret_key: str = field(default="", repr=False)
    host: str = "https://cloud.langfuse.com"
    environment: str = "development"
    release: str = "local"

    @classmethod
    def from_env(cls, *, required: bool = False):
        """Read the Langfuse settings. Raises ``RuntimeError`` when tracing is enabled without keys,
        or when `required` and tracing is disabled.
        """
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
            environment=os.getenv("LANGFUSE_ENVIRONMENT",
                                  "development").strip(),
            release=os.getenv("LANGFUSE_RELEASE", "local").strip(),
        )
        if required and not settings.enabled:
            raise RuntimeError(
                "Langfuse tracing must be enabled for live execution")
        if settings.enabled and (not settings.public_key or not settings.secret_key):
            raise RuntimeError(
                "Langfuse tracing credentials are not configured")
        return settings
