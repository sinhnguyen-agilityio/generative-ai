from abc import ABC, abstractmethod
from langchain_core.language_models import BaseChatModel


class LLMProvider(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def create(self) -> BaseChatModel:
        """Create and return a configured chat model."""
        ...
