from llms.base import LLMProvider
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings


class GenAIProvider(LLMProvider):
    def create(self):
        return ChatGoogleGenerativeAI(api_key=settings.genai_api_key, model=settings.genai_model)
