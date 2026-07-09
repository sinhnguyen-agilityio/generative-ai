from config import settings
from llm.client import LLMClient
from llm.classifier import SecureClassifier
from security.input_filter import InputFilter
from security.output_validation import OutputValidator
from security.risk_score import RiskScorer
from security.pipeline import SecureLLMPipeline

llm_client = LLMClient()
classifier = SecureClassifier(llm_client)

pipeline = SecureLLMPipeline(
    input_filter=InputFilter(),
    risk_scorer=RiskScorer(),
    classifier=classifier,
    output_validator=OutputValidator(),
)

while True:
    text = input("Enter text (or 'quit'): ")

    if text.lower() == "quit":
        break

    result = pipeline.process(text)
    print(result.model_dump_json(indent=2))
