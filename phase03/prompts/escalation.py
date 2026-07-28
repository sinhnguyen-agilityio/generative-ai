from langchain_core.prompts import ChatPromptTemplate

ESCALATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an experienced customer support analyst.

Your task is to determine whether this ticket should be escalated.

Instructions:
- Analyze the customer's issue.
- Consider the detected intent.
- Consider the detected sentiment.
- Consider the urgency level.
- Decide whether escalation is required.
- Select the most appropriate escalation team.
- Provide a short explanation.
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
