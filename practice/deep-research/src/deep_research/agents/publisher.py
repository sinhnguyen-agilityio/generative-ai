"""Publisher agent: saves the approved report as a Markdown file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from deep_research.states.research_state import ResearchState
from deep_research.utils.runtime import thread_id


class PublisherAgent:
    def __init__(self, output_dir: str):
        """Configure the destination for approved Markdown reports.

        Args:
            output_dir: Directory created on first publication.
        """
        self.output_dir = Path(output_dir)

    def publish(self, state: ResearchState, config: RunnableConfig):
        """Write the approved report to a timestamped Markdown file.

        Args:
            state: Research state containing the approved report.
            config: Graph configuration identifying the research thread.

        Returns:
            Published file path and an assistant message confirming publication.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{thread_id(config)[:8]}.md"
        path.write_text(state["report"], encoding="utf-8")
        return {"published_path": str(path), "messages": [AIMessage(f"Report published to `{path}`.")]}
