import json

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

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
