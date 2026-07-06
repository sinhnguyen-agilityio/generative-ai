from phase01.config import settings
from phase01.llm.client import LLMClient
from phase01.llm.classifier import SecureClassifier, StrangeSequence
from phase01.security.input_filter import InputFilter
from phase01.security.pipeline import SimpleSecurityPipeline

llm_client = LLMClient()
classifier = SecureClassifier(llm_client)

pipeline = SimpleSecurityPipeline(
    input_filter=InputFilter(),
    strangeSequence=StrangeSequence(llm_client)
)

while True:
    text = input("Enter text (or 'quit'): ")

    if text.lower() == "quit":
        break

    result = pipeline.process(text)
