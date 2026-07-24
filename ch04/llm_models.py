from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

import os
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")


def get_llm():
    return ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-5-nano")


def get_gemini():
    return ChatGoogleGenerativeAI(api_key=gemini_api_key, model="gemini-3.6-flash")
