from ddgs import DDGS

from models.search_result import SearchResult
from tools.base import BaseTool


class DuckDuckGoSearchTool(BaseTool):
    """Search the web using the DuckDuckGo search API."""
    name = "duckduckgo"
    description = "Search the web."

    def __init__(
        self,
        max_results: int = 3,
    ) -> None:
        """Store how many search results should be returned."""
        self.max_results = max_results

    def run(
        self,
        query: str,
    ) -> list[SearchResult]:
        """Run a search and return the results in a simple structured format."""
        with DDGS() as ddgs:
            results = ddgs.text(
                query=query,
                max_results=self.max_results,
            )

            output: list[SearchResult] = []

            for result in results:
                output.append(
                    SearchResult(
                        title=result.get("title", ""),
                        url=result.get("href", ""),
                        snippet=result.get("body", ""),
                    )
                )

            return output
