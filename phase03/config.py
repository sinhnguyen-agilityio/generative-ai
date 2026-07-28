from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Load environment settings used by the app."""

    def __init__(self) -> None:
        self.genai_api_key = os.getenv("GENAI_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.genai_model = os.getenv("GENAI_MODEL", "gemini-3.6-flash")
        self.force_openai_failure = bool(
            os.getenv("FORCE_OPENAI_FAILURE", False))
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        self.provider = os.getenv("PROVIDER", "openai")
        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")

        self.langfuse_host = os.getenv(
            "LANGFUSE_HOST", "https://cloud.langfuse.com")


settings = Settings()
