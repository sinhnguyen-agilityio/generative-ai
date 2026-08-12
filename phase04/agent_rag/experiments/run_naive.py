import asyncio

from rag.dataset import (
    build_evaluation_dataset,
    load_ragbench,
)
from rag.evaluation import (
    run_evaluation,
    save_results,
)
from rag.naive_rag import NaiveRAG


LIMIT = 10

OUTPUT_PATH = "reports/naive_rag_results.json"


async def main():

    print("Loading RAGBench...")
    dataset = load_ragbench()

    print(
        f"Preparing {LIMIT} evaluation samples..."
    )
    evaluation_dataset = build_evaluation_dataset(
        dataset,
        limit=LIMIT,
    )

    print("Initializing Naive RAG...")
    rag = NaiveRAG()

    print("Running evaluation...")
    results = await run_evaluation(
        rag,
        evaluation_dataset,
    )

    save_results(
        results,
        OUTPUT_PATH,
    )

    successful = sum(
        result["status"] == "success"
        for result in results
    )
    failed = len(results) - successful

    print("\n" + "=" * 60)
    print("Naive RAG evaluation completed")
    print("=" * 60)

    print(f"Total:      {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed:     {failed}")
    print(f"Output:     {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
