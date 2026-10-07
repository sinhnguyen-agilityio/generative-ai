"""Publisher agent: saves the approved report as a Markdown file."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from deep_research.agents.utils import thread_id
from deep_research.state import ResearchState


class PublisherAgent:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)

    def publish(self, state: ResearchState, config: RunnableConfig):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{thread_id(config)[:8]}.md"
        path.write_text(state["report"], encoding="utf-8")
        return {"published_path": str(path), "messages": [AIMessage(f"Report published to `{path}`.")]}
