import asyncio

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from evaluations.evaluator import RagasEvaluator
from generation.rag_chatbot import RAGChatbot
from ingestion.chunker import Chunker
from ingestion.loader import Loader
from ingestion.vector_store import VectorStore


load_dotenv()

# ----------------------------------------
# Configuration
# ----------------------------------------
RAG_CONCURRENCY = 5
EVALUATION_CONCURRENCY = 5

# ----------------------------------------
# Faithfulness scorer
# ----------------------------------------
client = AsyncOpenAI()
llm = llm_factory(
    "gpt-5-nano",
    client=client,
)
scorer = Faithfulness(llm=llm)
evaluation_semaphore = asyncio.Semaphore(
    EVALUATION_CONCURRENCY
)


async def score_sample(index, sample):
    async with evaluation_semaphore:
        contexts = sample.retrieved_contexts[:3]

        result = await scorer.ascore(
            user_input=sample.user_input,
            response=sample.response,
            retrieved_contexts=contexts,
        )

        return index, result.value


# ----------------------------------------
# Main
# ----------------------------------------
async def main():
    # ----------------------------------------
    # Load documents
    # ----------------------------------------
    loader = Loader()
    docs = loader.load_folder(
        "data/CilentoTouristInfo"
    )
    print(
        f"Loaded documents: {len(docs)}"
    )

    # ----------------------------------------
    # Chunk documents
    # ----------------------------------------
    chunker = Chunker()
    chunks = chunker.chunk_documents(
        docs
    )
    print(
        f"Created chunks: {len(chunks)}"
    )

    # ----------------------------------------
    # Vector store
    # ----------------------------------------
    vector_store = VectorStore()

    retriever = vector_store.store_docs(
        "tourist_collection",
        chunks,
    ).as_retriever(search_kwargs={"k": 4})

    # ----------------------------------------
    # Load evaluation dataset
    # ----------------------------------------
    evaluator = RagasEvaluator()
    testset = evaluator.load_eval_dataset(
        "evaluations/dataset.json"
    )
    print(
        f"Testset size: {len(testset.samples)}"
    )

    # ----------------------------------------
    # Create chatbot
    # ----------------------------------------
    chatbot = RAGChatbot(
        retriever=retriever
    )

    # ----------------------------------------
    # Run RAG
    # ----------------------------------------
    questions = [
        sample.user_input
        for sample in testset.samples
        if isinstance(sample.user_input, str)
    ]

    rag_results = await chatbot.abatch(
        questions,
        max_concurrency=RAG_CONCURRENCY,
    )

    # ----------------------------------------
    # Build Ragas evaluation dataset
    # ---------------------------------------
    evaluation_samples = []

    for sample, result in zip(
        testset.samples,
        rag_results,
    ):
        evaluation_samples.append(
            {
                "user_input": result["question"],
                "retrieved_contexts": [
                    doc.page_content
                    for doc in result["documents"]
                ],
                "response": result["answer"].content,
                "reference": sample.reference,
            }
        )

    evaluation_dataset = evaluator.create_evaluation_dataset(
        evaluation_samples
    )

    # ----------------------------------------
    # Evaluate Faithfulness
    # ----------------------------------------
    results = await asyncio.gather(
        *[
            score_sample(index, sample)
            for index, sample in enumerate(
                evaluation_dataset.samples
            )
        ]
    )

    # ----------------------------------------
    # Print results
    # ----------------------------------------
    scores = []

    for index, score in results:
        scores.append(score)
        sample = evaluation_dataset.samples[index]
        print(
            f"\n[{index + 1}] "
            f"Faithfulness: {score:.4f}"
        )
        print(
            f"Question: {sample.user_input}"
        )
        print(
            f"Response: {sample.response}"
        )

    # ----------------------------------------
    # Average
    # ----------------------------------------
    average = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )
    print("\n" + "=" * 50)
    print(
        f"Average Faithfulness: {average:.4f}"
    )
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
