from langchain_core.prompts import ChatPromptTemplate

SENTIMENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert in sentiment analysis.

Analyze the overall emotional tone of the customer's message.

Instructions:
- Classify the sentiment as Positive, Neutral, or Negative.
- Return a confidence score between 0.0 and 1.0.
- Consider the overall meaning of the message.
- Do not focus on individual words alone.
- Ignore masked PII placeholders.
            """.strip(),
        ),
        (
            "human",
            "{text}",
        ),
    ]
)
