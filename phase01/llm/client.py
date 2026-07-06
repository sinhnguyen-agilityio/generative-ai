from phase01.config import settings

import openai


class LLMClient:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.model
        self.temperature = settings.temperature
        self.max_output_tokens = settings.max_output_tokens

    def generate(self, messages: list) -> str:
        try:
            response = self.client.responses.create(
                model=self.model,
                input=messages,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens
            )
            print(response.output_text)

            return response.output_text
        except Exception as e:
            print(f"Error occurred: {e}")
            raise
