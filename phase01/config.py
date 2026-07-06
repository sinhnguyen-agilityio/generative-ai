from __future__ import annotations
import os
from dotenv import load_dotenv

# OpenAI configuration
MAX_OUTPUT_TOKENS = 200

# input validation configuration
MAX_INPUT_LENGTH = 5000
MIN_INPUT_LENGTH = 1
MAX_WORD_COUNT = 100

# configuration for fuzzy matching
TYPO_SIMILARITY_THRESHOLD = 0.85

# Control unicode normalization
UNICODE_NORMALIZATION_FORM = "NFKC"
REMOVE_CONTROL_CHARACTERS = True
CHECK_MIXED_SCRIPTS = True

# Risk thresholds
ALLOW_THRESHOLD = 30
REVIEW_THRESHOLD = 60
BLOCK_THRESHOLD = 80

PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"forget\s+(everything|all)",
    r"developer\s+mode",
    r"system\s+prompt",
    r"reveal\s+.*prompt",
    r"print\s+your\s+instructions",
    r"repeat\s+your\s+hidden",
    r"act\s+as",
    r"jailbreak",
    r"bypass",
    r"disable\s+safety",
    r"you\s+are\s+now",
]

PROTECTED_KEYWORDS = [
    "ignore",
    "instruction",
    "instructions",
    "system",
    "assistant",
    "developer",
    "prompt",
    "policy",
    "secret",
    "hidden",
]

SUSPICIOUS_OUTPUT_PATTERNS = [
    r"system prompt",
    r"developer message",
    r"chain of thought",
    r"hidden instruction",
    r"internal instruction",
]
RISK_SCORES = {
    "prompt_injection": 40,
    "typoglycemia": 20,
    "unicode_obfuscation": 20,
    "encoded_payload": 30,
    "long_input": 10,
    "length": 5,
}


load_dotenv()


class Settings:
    """Application configuration."""

    def __init__(self) -> None:
        self.openai_api_key = self._get_required_env("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
        self.max_output_tokens = int(
            os.getenv("OPENAI_MAX_OUTPUT_TOKENS", str(MAX_OUTPUT_TOKENS)))

    @staticmethod
    def _get_required_env(name: str) -> str:
        """Read a required environment variable."""
        value = os.getenv(name)

        if not value:
            raise ValueError(f"Missing required environment variable: {name}")

        return value


settings = Settings()
