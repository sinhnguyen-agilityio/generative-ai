from langchain_openai import ChatOpenAI

from .config import LLM_MODEL
from .retriever import get_retriever


SYSTEM_PROMPT = """
You are a helpful question-answering assistant.

Answer the user's question using only the provided context.

If the context does not contain enough information,
say that you do not have enough information.

Do not invent facts that are not supported by the context.
"""


class NaiveRAG:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
        )
        self.retriever = get_retriever()

    async def ainvoke(
        self,
        question: str,
    ) -> dict:
        documents = await self.retriever.ainvoke(
            question
        )
        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        prompt = f"""
            {SYSTEM_PROMPT}

            Context:
            {context}

            Question:
            {question}

            Answer:
        """

        response = await self.llm.ainvoke(
            prompt
        )

        return {
            "answer": response.content,
            "documents": documents,
            "retrieval_calls": 1,
        }
