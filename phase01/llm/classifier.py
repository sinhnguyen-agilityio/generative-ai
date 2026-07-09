
from llm.prompts import build_classification_messages
from schemas import ClassificationResult, PipelineAction
from llm.client import LLMClient
from llm.strange import build_logic_messages
import json


class SecureClassifier:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def classify(self, text: str) -> ClassificationResult:
        # Implementation for classifying text
        messages = build_classification_messages(text)
        response = self.llm_client.generate(messages)
        # Parse the response and return a ClassificationResult
        return ClassificationResult.model_validate(
            json.loads(response)
        )


class StrangeSequence:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def analyze(self, text: str) -> str:
        # Implementation for analyzing strange sequences
        messages = build_logic_messages(text)
        response = self.llm_client.generate(messages)
        # Parse the response and return a ClassificationResult
        return response
