from config import settings
from llm.client import LLMClient
from llm.classifier import SecureClassifier, StrangeSequence
from security.input_filter import InputFilter
from security.pipeline import SimpleSecurityPipeline

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
    print(result)
