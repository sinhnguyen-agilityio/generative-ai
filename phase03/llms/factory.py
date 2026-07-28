from typing import Type

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from llms.openai_provider import OpenAIProvider
from llms.gemini_provider import GenAIProvider


class LLMFactory:
    """
    Factory responsible for creating LLM instances.

    Responsibilities:
    - Create provider instances
    - Apply structured output
    - Configure fallback models
    """

    @staticmethod
    def create(provider: str = "openai") -> BaseChatModel:
        """
        Create a chat model without structured output.
        """
        match provider.lower():
            case "openai":
                return OpenAIProvider().create()
            case "gemini":
                return GenAIProvider().create()
            case _:
                raise ValueError(f"Unsupported provider: {provider}")

    @classmethod
    def structured(
        cls,
        schema: Type[BaseModel],
        primary: str = "openai",
        fallback: str | None = "gemini",
    ):
        """
        Create a structured-output LLM with optional fallback.
        """

        primary_llm = cls.create(primary).with_structured_output(schema)

        if fallback is None:
            return primary_llm

        fallback_llm = cls.create(fallback).with_structured_output(schema)

        return primary_llm.with_fallbacks(
            [fallback_llm]
        )
