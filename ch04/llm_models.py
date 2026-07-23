from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def get_llm():
    return ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-5-nano")
