from __future__ import annotations
import os
from dotenv import load_dotenv

MODEL = "gpt-5.4-mini"

load_dotenv()


class Settings:
    """Application configuration."""

    def __init__(self) -> None:
        self.openai_api_key = self._get_required_env("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))

    @staticmethod
    def _get_required_env(name: str) -> str:
        """Read a required environment variable."""
        value = os.getenv(name)

        if not value:
            raise ValueError(f"Missing required environment variable: {name}")

        return value


settings = Settings()
