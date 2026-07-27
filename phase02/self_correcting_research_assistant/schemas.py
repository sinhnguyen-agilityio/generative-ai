from typing import List
from pydantic import BaseModel, Field


class Source(BaseModel):
    """ Schema for a source used by the agent """
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """ Schema for the agent response with answers and sources """
    answer: str = Field(
        description="The agent's answer to the user's question")
    sources: List[Source] = Field(
        description="The list of sources used by the agent to generate answer of the question", default_factory=list)
