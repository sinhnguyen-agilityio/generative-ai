from langchain_core.prompts import ChatPromptTemplate

# Prompt template used by the agent to decide whether to search again or answer.
research_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a self-correcting research assistant.

            Rules:
            - Think step by step.
            - If you need more evidence, use the action "duckduckgo" with a focused search query.
            - After each action, use the search result as the Observation.
            - If you already know the answer, provide the Final Answer directly.
            - Do not invent facts.
            - Keep the response concise and evidence-based.

            Search Results:
            {context}
            """,
        ),
        ("human", "{question}"),
    ]
)
