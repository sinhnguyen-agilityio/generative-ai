from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .config import LLM_MODEL
from .tool import DocumentSearcher


SYSTEM_PROMPT = """
You are a helpful question-answering assistant.

You have access to a search tool for the knowledge base.

When answering:

1. Use the search tool when you need information
   from the knowledge base.
2. You may search more than once if the first search
   does not provide enough information.
3. Base your answer on retrieved information.
4. Do not invent facts.
5. If the information is insufficient, say so.
6. Give a concise and accurate final answer.
"""


class AgentRAG:

    def __init__(self):
        self.searcher = DocumentSearcher()
        self.agent = create_agent(
            model=ChatOpenAI(
                model=LLM_MODEL,
                temperature=0,
            ),
            tools=[
                self.searcher.tool,
            ],
            system_prompt=SYSTEM_PROMPT,
        )

    async def ainvoke(
        self,
        question: str,
    ) -> dict:
        # Reset state for each question
        self.searcher.reset()
        result = await self.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            }
        )
        messages = result["messages"]
        # Find the final AI response
        answer = ""

        for message in reversed(messages):
            if message.type == "ai":
                content = message.content
                if isinstance(content, str):
                    answer = content
                    break

        return {
            "answer": answer,
            "documents": self.searcher.documents,
            "retrieval_calls": self.searcher.calls,
        }
