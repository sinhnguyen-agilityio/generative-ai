from typing import Any, Dict, Iterator

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage

from llm import get_llm
from prompt import REACT_PROMPT
from tool import get_tools
from trace import get_langfuse_handler


class ResearchAgent:
    """Self-Correcting Research Assistant."""

    def __init__(self, recursion_limit: int = 10):

        self.handler = get_langfuse_handler()
        self.recursion_limit = recursion_limit
        self.agent = create_agent(
            model=get_llm(),
            tools=get_tools(),
            system_prompt=REACT_PROMPT,
        )

    def invoke(self, question: str) -> str:
        """
        Execute the agent and return only the final answer.
        """
        response = self.agent.invoke(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            config={"callbacks": [self.handler], "recursion_limit": self.recursion_limit}
        )

        return response["messages"][-1].content

    def stream(self, question: str) -> Iterator[Dict[str, Any]]:
        """
        Stream every execution step from the agent.

        This is useful for:
        - ReAct visualization
        - Langfuse tracing
        - Debugging
        """

        yield from self.agent.stream(
            {
                "messages": [
                    HumanMessage(content=question)
                ]
            },
            stream_mode="updates",
            config={"callbacks": [self.handler], "recursion_limit": self.recursion_limit}
        )


def build_agent() -> ResearchAgent:
    return ResearchAgent()
