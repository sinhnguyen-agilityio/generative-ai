from langchain_openai import ChatOpenAI
from config import settings


def get_llm():
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        model="gpt-5.4-nano",
        temperature=0,
    )
