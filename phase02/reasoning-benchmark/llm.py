from __future__ import annotations

from langfuse.openai import OpenAI
from models.llm_response import LLMResponse
from config import settings


class LLMClient:
    """
    Wrapper around the OpenAI Responses API.

    The rest of the application should never directly access
    the OpenAI SDK.
    """

    def __init__(
        self
    ) -> None:

        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.model

    def generate(
        self,
        prompt: str,
        
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.
        """

        response = self.client.responses.create(
            model=self.model,
            input=prompt,
        )

        usage = response.usage

        return LLMResponse(
            text=response.output_text,
            model=response.model,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )
