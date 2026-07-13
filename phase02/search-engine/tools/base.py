from abc import ABC, abstractmethod
from models.search_result import SearchResult


class BaseTool(ABC):
    """Base class for tools that the agent can use to gather information."""

    name: str
    description: str

    @abstractmethod
    def run(
        self,
        query: str,
    ) -> list[SearchResult]:
        """Execute the tool with a search query and return structured results."""
        raise NotImplementedError
