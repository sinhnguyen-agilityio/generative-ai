import json

from langfuse.openai import OpenAI
from config import settings


class LLM:
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(
            model=settings.model,
            input=prompt,
        )

        return response.output_text

    def generate_json(self, prompt: str):
        text = self.generate(prompt)

        return json.loads(text)
