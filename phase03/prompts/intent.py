from langchain_core.prompts import ChatPromptTemplate

INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert in customer support intent classification.

Determine the customer's primary intent.

Instructions:
- Choose the single best intent category.
- Return a confidence score between 0.0 and 1.0.
- Base your decision only on the provided message.
- Do not infer information that is not explicitly stated.
- Ignore masked PII placeholders such as <EMAIL>, <PHONE_NUMBER>, and <CREDIT_CARD>.
            """.strip(),
        ),
        (
            "human",
            "{text}",
        ),
    ]
)
