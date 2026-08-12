import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from ragas.llms import llm_factory
from ragas.metrics.collections import (
    Faithfulness,
    ContextRecall,
)

from dotenv import load_dotenv

load_dotenv()


NAIVE_PATH = "reports/naive_rag_results.json"
AGENT_PATH = "reports/agent_rag_results.json"

NAIVE_SCORE_PATH = "reports/naive_rag_scores.json"
AGENT_SCORE_PATH = "reports/agent_rag_scores.json"


def load_results(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_results(
    results: list[dict],
    path: str,
) -> None:

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )


async def evaluate_results(
    results: list[dict],
    evaluator_llm,
) -> list[dict]:

    faithfulness = Faithfulness(
        llm=evaluator_llm,
    )

    context_recall = ContextRecall(
        llm=evaluator_llm,
    )

    scores = []

    for index, row in enumerate(
        results,
        start=1,
    ):

        print(
            f"[{index}/{len(results)}] "
            f"{row['question']}"
        )

        if row.get("status") != "success":
            continue

        try:

            # 1. Input dành cho Faithfulness
            faithfulness_input = {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": row["contexts"],
            }

            faithfulness_score = (
                await faithfulness.ascore(
                    **faithfulness_input
                )
            )

            # 2. Input dành cho Context Recall
            context_recall_input = {
                "user_input": row["question"],
                "retrieved_contexts": row["contexts"],
                "reference": row["ground_truth"],
            }

            context_recall_score = (
                await context_recall.ascore(
                    **context_recall_input
                )
            )

            # 3. Save
            scores.append({
                "id": row["id"],
                "question": row["question"],
                "faithfulness": (
                    faithfulness_score.value
                ),
                "context_recall": (
                    context_recall_score.value
                ),
                "retrieval_calls": row.get(
                    "retrieval_calls",
                    0,
                ),
                "status": "success",
                "error": None,
            })

        except Exception as exc:

            print(f"ERROR: {exc}")

            scores.append({
                "id": row["id"],
                "question": row["question"],
                "faithfulness": None,
                "context_recall": None,
                "retrieval_calls": row.get(
                    "retrieval_calls",
                    0,
                ),
                "status": "error",
                "error": str(exc),
            })

    return scores


async def main():

    print("=" * 70)
    print("RAGAS EVALUATION")
    print("=" * 70)

    # --------------------------------------------------
    # 1. Load generated RAG results
    # --------------------------------------------------

    print("\nLoading Naive RAG results...")

    naive_results = load_results(
        NAIVE_PATH
    )

    print(
        f"Naive RAG samples: "
        f"{len(naive_results)}"
    )

    print("\nLoading Agent RAG results...")

    agent_results = load_results(
        AGENT_PATH
    )

    print(
        f"Agent RAG samples: "
        f"{len(agent_results)}"
    )

    # --------------------------------------------------
    # 2. Create evaluator LLM
    # --------------------------------------------------

    print("\nInitializing evaluator LLM...")

    client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client,
    )

    # --------------------------------------------------
    # 3. Evaluate Naive RAG
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("Evaluating Naive RAG")
    print("=" * 70)

    naive_scores = await evaluate_results(
        naive_results,
        evaluator_llm,
    )

    # --------------------------------------------------
    # 4. Evaluate Agent RAG
    # --------------------------------------------------

    print("\n")
    print("=" * 70)
    print("Evaluating Agent RAG")
    print("=" * 70)

    agent_scores = await evaluate_results(
        agent_results,
        evaluator_llm,
    )

    # --------------------------------------------------
    # 5. Save scores
    # --------------------------------------------------

    save_results(
        naive_scores,
        NAIVE_SCORE_PATH,
    )

    save_results(
        agent_scores,
        AGENT_SCORE_PATH,
    )

    # --------------------------------------------------
    # 6. Summary
    # --------------------------------------------------

    naive_success = sum(
        x["status"] == "success"
        for x in naive_scores
    )

    agent_success = sum(
        x["status"] == "success"
        for x in agent_scores
    )

    print("\n")
    print("=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)

    print(
        f"\nNaive RAG:"
        f"\n  Samples: {len(naive_scores)}"
        f"\n  Success: {naive_success}"
    )

    print(
        f"\nAgent RAG:"
        f"\n  Samples: {len(agent_scores)}"
        f"\n  Success: {agent_success}"
    )

    print("\nOutput files:")

    print(
        f"  {NAIVE_SCORE_PATH}"
    )

    print(
        f"  {AGENT_SCORE_PATH}"
    )


if __name__ == "__main__":
    asyncio.run(main())
