import json
import csv
from pathlib import Path


NAIVE_PATH = "reports/naive_rag_scores.json"
AGENT_PATH = "reports/agent_rag_scores.json"
OUTPUT_PATH = "reports/rag_comparison.csv"


def load_scores(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def average(
    values: list[float],
) -> float:

    if not values:
        return 0.0

    return sum(values) / len(values)


def calculate_metrics(
    results: list[dict],
) -> dict:

    valid_results = [
        row
        for row in results
        if row.get("status") == "success"
    ]

    faithfulness = [
        row["faithfulness"]
        for row in valid_results
        if row.get("faithfulness") is not None
    ]

    context_recall = [
        row["context_recall"]
        for row in valid_results
        if row.get("context_recall") is not None
    ]

    retrieval_calls = [
        row["retrieval_calls"]
        for row in valid_results
    ]

    return {
        "samples": len(valid_results),
        "faithfulness": average(
            faithfulness
        ),
        "context_recall": average(
            context_recall
        ),
        "avg_retrieval_calls": average(
            retrieval_calls
        ),
    }


def main():

    print("Loading scores...")

    naive_results = load_scores(
        NAIVE_PATH
    )

    agent_results = load_scores(
        AGENT_PATH
    )

    naive = calculate_metrics(
        naive_results
    )

    agent = calculate_metrics(
        agent_results
    )

    comparison = [
        {
            "metric": "Faithfulness",
            "naive_rag": naive["faithfulness"],
            "agent_rag": agent["faithfulness"],
        },
        {
            "metric": "Context Recall",
            "naive_rag": naive["context_recall"],
            "agent_rag": agent["context_recall"],
        },
        {
            "metric": "Avg Retrieval Calls",
            "naive_rag": naive["avg_retrieval_calls"],
            "agent_rag": agent["avg_retrieval_calls"],
        },
    ]

    Path("reports").mkdir(
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "naive_rag",
                "agent_rag",
            ],
        )

        writer.writeheader()

        writer.writerows(comparison)

    print("\n" + "=" * 70)
    print("RAG COMPARISON")
    print("=" * 70)

    print(
        f"\nSamples:"
        f"\n  Naive RAG: {naive['samples']}"
        f"\n  Agent RAG: {agent['samples']}"
    )

    print("\nMetrics:")

    print(
        f"\nFaithfulness"
        f"\n  Naive RAG : {naive['faithfulness']:.4f}"
        f"\n  Agent RAG : {agent['faithfulness']:.4f}"
        f"\n  Difference: "
        f"{agent['faithfulness'] - naive['faithfulness']:+.4f}"
    )

    print(
        f"\nContext Recall"
        f"\n  Naive RAG : {naive['context_recall']:.4f}"
        f"\n  Agent RAG : {agent['context_recall']:.4f}"
        f"\n  Difference: "
        f"{agent['context_recall'] - naive['context_recall']:+.4f}"
    )

    print(
        f"\nAvg Retrieval Calls"
        f"\n  Naive RAG : {naive['avg_retrieval_calls']:.2f}"
        f"\n  Agent RAG : {agent['avg_retrieval_calls']:.2f}"
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
