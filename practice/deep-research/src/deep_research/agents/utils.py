"""Helpers shared by the agents."""

from __future__ import annotations

from datetime import datetime

from langchain_core.runnables import RunnableConfig


def today() -> str:
    return datetime.now().strftime("%a %b %d, %Y")


def bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def thread_id(config: RunnableConfig) -> str:
    return config["configurable"]["thread_id"]
