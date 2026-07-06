from phase01.config import settings
from phase01.llm.client import LLMClient
from phase01.llm.classifier import SecureClassifier
from phase01.security.input_filter import InputFilter
from phase01.security.output_validation import OutputValidator
from phase01.security.risk_score import RiskScorer
from phase01.security.pipeline import SecureLLMPipeline

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
