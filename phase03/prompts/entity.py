from langchain_core.prompts import ChatPromptTemplate

ENTITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert in information extraction.

Extract business-related entities explicitly mentioned in the customer message.

Supported entities:
- Customer name
- Product names
- Order IDs
- Ticket IDs
- Account IDs
- Dates

Instructions:
- Extract only values explicitly mentioned.
- Do not infer or generate missing values.
- Ignore masked placeholders such as <EMAIL>, <PHONE_NUMBER>, and <CREDIT_CARD>.
- Return empty lists when no entities are found.
            """.strip(),
        ),
        (
            "human",
            "{text}",
        ),
    ]
)
