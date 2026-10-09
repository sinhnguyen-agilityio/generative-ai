"""Read research thread identifiers from graph configuration."""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig


def thread_id(config: RunnableConfig) -> str:
    """Read the research thread identifier from graph configuration.

    Args:
        config: Runnable configuration containing configurable.thread_id.

    Returns:
        Thread identifier used for storage and tracing.
    """
    return config["configurable"]["thread_id"]
