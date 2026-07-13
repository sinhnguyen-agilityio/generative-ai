from dataclasses import dataclass


@dataclass(slots=True)
class SearchResult:
    title: str
    url: str
    snippet: str
