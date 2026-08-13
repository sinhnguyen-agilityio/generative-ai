from langchain_core.prompts import ChatPromptTemplate

URGENCY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an experienced customer support analyst.

Your task is to determine the urgency of a customer support ticket.

Instructions:
- Analyze the customer's issue.
- Consider the customer's intent and sentiment.
- Determine the urgency level.
- Provide a short explanation for your decision.
- Use only the information contained in the analysis report.
- Do not invent information.
            """.strip(),
        ),
        (
            "human",
            "{report}",
        ),
    ]
)
