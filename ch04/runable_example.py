from langchain_core.runnables import RunnableLambda
from llm_models import get_llm, get_gemini
import random


def add_one(x: int) -> int:
    return x + 1


def buggy_double(y: int) -> int:
    """Buggy code that will fail 70% of the time"""
    if random.random() > 0.3:
        print('This code failed, and will probably be retried!')  # noqa: T201
        raise ValueError('Triggered buggy code')
    return y * 2


sequence = (
    RunnableLambda(add_one) |
    RunnableLambda(buggy_double).with_retry(  # Retry on failure
        stop_after_attempt=10,
        wait_exponential_jitter=False
    )
)

# print(sequence.batch([1, 2, 4]))
# print(sequence.invoke(1))

openai = get_llm()
gemini = get_gemini()

response = openai.invoke("What is Python?")
print(response.content_blocks[0].get('text'))


response1 = gemini.invoke("What is Python?")
print(response1.content_blocks[0].get('text'))
