import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

from generation.rag_chatbot import RAGChatbot
from openai import OpenAI


class RagasEvaluator:
    def __init__(self):
        self.generator_llm = LangchainLLMWrapper(
            ChatOpenAI(model="gpt-5-nano"))
        openai_client = OpenAI()
        self.generator_embeddings = OpenAIEmbeddings(client=openai_client)

    def gen_testset(
        self,
        docs,
        testset_size: int = 10,
    ):
        generator = TestsetGenerator(
            llm=self.generator_llm, embedding_model=self.generator_embeddings)
        docs = list(docs)

        return generator.generate_with_langchain_docs(docs, testset_size)

    def load_eval_dataset(self, path: str) -> EvaluationDataset:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return EvaluationDataset.from_list(data)

    def create_evaluation_dataset(
        self,
        samples: list[dict],
    ) -> EvaluationDataset:
        return EvaluationDataset.from_list(samples)

    async def run_rag_testset(
        self,
        chatbot: RAGChatbot,
        testset,
        max_concurrency: int = 5,
    ) -> EvaluationDataset:
        samples = [
            sample
            for sample in testset.samples
            if isinstance(sample.user_input, str)
        ]
        questions = [
            sample.user_input
            for sample in samples
        ]
        # Run RAG concurrently
        results = await chatbot.abatch(
            questions,
            max_concurrency=max_concurrency,
        )

        evaluation_samples = []

        for sample, result in zip(
            samples,
            results,
        ):
            evaluation_samples.append(
                {
                    "user_input": sample.user_input,
                    "retrieved_contexts": [
                        document.page_content
                        for document in result["documents"]
                    ],
                    "response": result["answer"].content,
                    "reference": sample.reference,
                }
            )

        return EvaluationDataset.from_list(
            evaluation_samples
        )
