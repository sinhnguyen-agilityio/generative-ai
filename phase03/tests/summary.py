from llms.factory import LLMFactory
from chains.summary import build
from models.ticket import SecureTicket


llm = LLMFactory.create()

chain = build(llm)

ticket = SecureTicket(
    original_text="I cannot login to my account.",
    sanitized_text="I cannot login to my account.",
)

result = chain.invoke(ticket)

print(result)
