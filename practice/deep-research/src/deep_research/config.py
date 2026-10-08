"""Settings read from the environment. A local .env is loaded without overriding the process."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _flag(name: str, default: bool = False) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ModelSpec:
    """One model tier: a primary model and an optional fallback (run with default settings)."""

    model: str
    provider: str | None = None
    kwargs: dict = field(default_factory=dict)
    fallback_model: str | None = None
    fallback_provider: str | None = None
    timeout: float = 60.0

    @classmethod
    def from_env(cls, tier: str, timeout: float) -> ModelSpec:
        prefix = f"INTELLIGENCE_{tier}_"
        model = _env(prefix + "MODEL")
        if not model:
            raise ValueError(f"{prefix}MODEL is required")
        return cls(
            model=model,
            provider=_env(prefix + "PROVIDER") or None,
            kwargs=json.loads(_env(prefix + "KWARGS") or "{}"),
            fallback_model=_env(prefix + "FALLBACK_MODEL") or None,
            fallback_provider=_env(prefix + "FALLBACK_PROVIDER") or None,
            timeout=float(_env(prefix + "TIMEOUT") or timeout),
        )


@dataclass(frozen=True)
class Settings:
    reasoning: ModelSpec
    standard: ModelSpec
    tavily_url: str = "https://mcp.tavily.com/mcp/"
    tavily_api_key: str = field(default="", repr=False)
    chroma_path: str = ".data/chroma"
    embedding_model: str = "gemini-embedding-2"
    embedding_provider: str = "google_genai"
    evaluation_model: str = "gemini-flash-latest"  # RAGAS judge, a Gemini model (GOOGLE_API_KEY)
    checkpoint_path: str = ".data/checkpoints.sqlite3"
    output_dir: str = "exports"
    max_concurrent_researchers: int = 3
    max_researcher_searches: int = 5
    retrieval_k: int = 6  # passages of the result pages shown to the researcher per search
    # Fixed limits, not configurable from the environment.
    max_supervisor_iterations: int = 6
    search_max_results: int = 5
    langfuse_enabled: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv(override=False)
        return cls(
            reasoning=ModelSpec.from_env("REASONING", timeout=120),
            standard=ModelSpec.from_env("STANDARD", timeout=60),
            tavily_url=_env("TAVILY_MCP_URL", cls.tavily_url),
            tavily_api_key=_env("TAVILY_API_KEY"),
            chroma_path=_env("INTELLIGENCE_CHROMA_PATH", cls.chroma_path),
            embedding_model=_env("INTELLIGENCE_EMBEDDING_MODEL", cls.embedding_model),
            embedding_provider=_env("INTELLIGENCE_EMBEDDING_PROVIDER", cls.embedding_provider),
            evaluation_model=_env("INTELLIGENCE_EVALUATION_MODEL", cls.evaluation_model),
            checkpoint_path=_env("INTELLIGENCE_CHECKPOINT_PATH", cls.checkpoint_path),
            output_dir=_env("INTELLIGENCE_OUTPUT_DIR", cls.output_dir),
            max_concurrent_researchers=int(_env("INTELLIGENCE_MAX_CONCURRENT_RESEARCHERS", "3")),
            max_researcher_searches=int(_env("INTELLIGENCE_MAX_SEARCH_CALLS", "5")),
            retrieval_k=int(_env("INTELLIGENCE_RETRIEVAL_K", "6")),
            langfuse_enabled=_flag("INTELLIGENCE_LANGFUSE_ENABLED"),
        )
