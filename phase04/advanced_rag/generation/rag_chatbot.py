import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import ChatOpenAI
from retrieval.query_rewriter import rewriter_chain
from retrieval.security import SecureContextProcessor
from tracer import langfuse_handler, langfuse
from utils.pii import PIIMasker


class RAGChatbot:
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are a helpful assistant, world-class
                expert in Roman and Greek history, especially in towns
                located in southern Italy. Provide interesting insights
                on local history and recommend places to visit with
                knowledgeable and engaging answers. Answer all questions
                to the best of your ability, but only use what has been
                provided in the context. If you don't know, just say
                you don't know. Use three sentences maximum and keep
                the answer as concise as possible.
                """,
            ),
            ("assistant", "{retrieved_context}"),
            ("human", "{question}"),
        ]
    )

    def __init__(
        self,
        retriever: VectorStoreRetriever,
        model: str = "gpt-5-nano",

    ):
        self.retriever = retriever
        self.chatbot = ChatOpenAI(model=model)

    async def _rewrite_question(self, question: str) -> str:
        rewritten = await rewriter_chain.ainvoke(
            {"user_question": question}
        )

        return rewritten.strip().strip('"')

    async def _retrieve(self, question: str):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="rag_chatbot",
            input={
                "question": question
            }
        ) as similarity_search:
            rewritten_question = await self._rewrite_question(question)
            documents = await self.retriever.ainvoke(
                rewritten_question
            )

            similarity_search.update(output={
                "rewritten_question": rewritten_question,
                "documents": len(documents)
            })

            return documents

    async def ainvoke(self, question: str):
        with langfuse.start_as_current_observation(
            as_type="span",
            name="rag_chatbot",
        ) as rag_chatbot:
            pii_masker = PIIMasker()
            secure_processor = SecureContextProcessor(
                pii_masker
            )
            documents = await self._retrieve(question)

            safe_docs = secure_processor.process(
                documents
            )

            answer = await (
                self.rag_prompt | self.chatbot
            ).ainvoke(
                {
                    "question": question,
                    "retrieved_context": safe_docs,
                },
                config={"callbacks": [langfuse_handler]}
            )

            rag_chatbot.update(output={
                "question": question,
                "answer": answer
            })

            return {
                "question": question,
                "answer": answer,
                "documents": safe_docs,
            }

    async def abatch(
        self,
        questions: list[str],
        max_concurrency: int = 1,
    ):
        """
        Execute multiple questions concurrently.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process(question: str):
            async with semaphore:
                return await self.ainvoke(question)

        return await asyncio.gather(
            *(process(question) for question in questions)
        )
