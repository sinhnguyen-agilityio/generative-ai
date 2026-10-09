"""Settings definitions."""

from __future__ import annotations

from dataclasses import dataclass, field

from dotenv import load_dotenv

from deep_research.model_spec import ModelSpec
from deep_research.utils.env import _env, _flag


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
        """Load application settings without overriding existing environment variables.

        Returns:
            Settings for model tiers, search, storage, limits, and tracing.
        """
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
