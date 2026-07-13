from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Load environment settings used by the app."""

    def __init__(self) -> None:
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        self.langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")
        self.langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
        self.langfuse_host = os.getenv(
            "LANGFUSE_HOST", "https://cloud.langfuse.com")


settings = Settings()
