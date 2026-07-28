from llms.base import LLMProvider
from langchain_openai import ChatOpenAI
from config import settings


class OpenAIProvider(LLMProvider):

    def create(self):
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            max_retries=2
        )
