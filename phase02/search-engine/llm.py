from langchain_openai import ChatOpenAI
from config import settings


class OpenAIClient:
    """Small wrapper that creates and calls the OpenAI chat model."""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            temperature=settings.temperature,
        )

    def invoke(self, prompt: str, callbacks=None):
        """Send a prompt to the LLM and optionally attach callbacks for tracing."""
        if callbacks:
            return self.llm.invoke(prompt, config={"callbacks": callbacks})
        return self.llm.invoke(prompt)
