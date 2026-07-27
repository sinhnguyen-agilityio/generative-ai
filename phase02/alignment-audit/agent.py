from __future__ import annotations

from collections.abc import Iterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

from callbacks import get_langfuse_handler
from config import settings
from langchain_community.tools import TavilySearchResults

web_search = TavilySearchResults(max_results=2)

TOOLS = [web_search]


class LangChainAgent:
    """
    Thin wrapper around LangChain's create_agent().
    Responsible only for executing the agent and
    normalizing streamed events.
    """

    def __init__(self):

        self.llm = ChatOpenAI(
            model=settings.model,
            temperature=0,
        )

        self.agent = create_agent(
            model=self.llm,
            tools=TOOLS,
        )

    def execute(
        self,
        prompt: str,
    ) -> Iterator[dict]:
        stream = self.agent.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            },
            config={
                "callbacks": [
                    get_langfuse_handler(),
                ]
            },
        )

        #
        # Normalize stream output so detector.py
        # doesn't care about LangChain internals.
        #
        for event in stream:
            yield self._normalize_event(event)

    def _normalize_event(
        self,
        event: dict,
    ) -> dict:

        #
        # We'll update this after inspecting
        # the actual event schema.
        #

        if "messages" in event:
            return {
                "agent": {
                    "messages": event["messages"],
                }
            }

        return {
            "agent": {
                "messages": [
                    AIMessage(
                        content=str(event),
                    )
                ]
            }
        }
