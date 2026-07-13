from dataclasses import dataclass, field
from models.react_step import ReActStep
from models.search_result import SearchResult


@dataclass(slots=True)
class AgentResponse:
    """
    Standard response returned by every agent.

    This model will grow as we implement more capabilities.
    """
    answer: str
    search_results: list[SearchResult] = field(default_factory=list)
    thoughts: list[str] = field(default_factory=list)
    steps: list[ReActStep] = field(default_factory=list)
    iterations: int = 0
