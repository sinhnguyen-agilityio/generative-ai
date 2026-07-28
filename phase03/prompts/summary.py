from langchain_core.prompts import ChatPromptTemplate

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
                You are an experienced customer support analyst.

                Your task is to summarize the customer's issue.

                Instructions:
                - Write a concise summary in 1-2 sentences.
                - Preserve the customer's original intent.
                - Use only the information explicitly stated in the message.
                - Do not infer or invent missing information.
                - Ignore masked PII placeholders such as <EMAIL>, <PHONE_NUMBER>, and <CREDIT_CARD>.
            """.strip(),
        ),
        (
            "human",
            "{text}",
        ),
    ]
)
