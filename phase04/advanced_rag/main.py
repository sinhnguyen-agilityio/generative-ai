from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.storage import InMemoryStore
import asyncio
from datetime import datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas import Dataset, experiment
from ragas.backends.inmemory import InMemoryBackend
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness

from evaluations.evaluator import RagasEvaluator
from generation.rag_chatbot import RAGChatbot
from ingestion.chunker import Chunker
from ingestion.loader import Loader
from ingestion.vector_store import VectorStore
from pathlib import Path


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
llm = llm_factory("gpt-4o-mini", client=client)

scorer = Faithfulness(llm=llm)
evaluation_semaphore = asyncio.Semaphore(
    EVALUATION_CONCURRENCY
)


def _get_question_from_row(row):
    if isinstance(row, dict):
        return row.get("user_input") or row.get("question") or row.get("input")

    for field in ("user_input", "question", "input"):
        value = getattr(row, field, None)
        if value is not None:
            return value

    return None


def create_baseline_experiment(chatbot: RAGChatbot):
    @experiment()
    async def baseline_experiment(row):
        question = _get_question_from_row(row)
        if not question:
            raise ValueError(f"Missing question in row: {row}")

        result = await chatbot.ainvoke(question)
        answer = result["answer"]
        response = answer.content if hasattr(
            answer, "content") else str(answer)

        documents = result.get("documents", [])
        retrieved_contexts = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in documents
        ]

        faithfulness = await scorer.ascore(
            user_input=question,
            response=response,
            retrieved_contexts=retrieved_contexts
        )

        base_row = row.model_dump(exclude_none=True) if hasattr(
            row, "model_dump") else dict(row)

        return {
            **base_row,
            "response": response,
            "faithfulness": faithfulness.value,
            "retrieved_contexts": retrieved_contexts,
            "experiment_name": "baseline_v1",
            "model_version": "gpt-4o-mini",
            "timestamp": datetime.now().isoformat(),
        }

    return baseline_experiment


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
    from langchain_classic.retrievers import ParentDocumentRetriever

    parent_splitter = Chunker(chunk_size=3000).text_splitter
    child_splitter = Chunker(chunk_size=500).text_splitter

    child_chunks_collection = Chroma(
        collection_name="tourist_child_chunks",
        embedding_function=OpenAIEmbeddings(),
    )

    child_chunks_collection.reset_collection()
    doc_store = InMemoryStore()

    parent_doc_retriever = ParentDocumentRetriever(
        vectorstore=child_chunks_collection,
        docstore=doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={
            "k": 4,
        },

    )

    parent_doc_retriever.add_documents(docs)

    # ----------------------------------------
    # Vector store
    # ----------------------------------------
    # vector_store = VectorStore()

    # retriever = vector_store.store_docs(
    #     "tourist_collection",
    #     chunks,
    # ).as_retriever(search_kwargs={"k": 4})

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
        retriever=parent_doc_retriever
    )

    # ----------------------------------------
    # Run a Ragas experiment against the dataset rows
    # ----------------------------------------
    experiment_dataset = Dataset(
        "baseline_eval",
        backend=InMemoryBackend(),
        data=[
            {"user_input": sample.user_input, "reference": sample.reference}
            for sample in testset.samples
            if isinstance(sample.user_input, str)
        ],
    )

    baseline_experiment = create_baseline_experiment(chatbot)
    experiment_results = await baseline_experiment.arun(
        experiment_dataset,
        name="baseline_v1",
    )
    print(f"Experiment rows: {len(experiment_results)}")
    df = experiment_results.to_pandas()
    Path("reports").mkdir(parents=True, exist_ok=True)

    df.to_csv(
        "reports/ragas_experiment_report.csv",
        index=False,
    )

if __name__ == "__main__":
    asyncio.run(main())
