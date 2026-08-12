import json
from pathlib import Path
from typing import Any


def serialize_documents(documents) -> list[str]:
    return [
        document.page_content
        for document in documents
    ]


def save_results(
    results: list[dict[str, Any]],
    output_path: str,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            ensure_ascii=False,
            indent=2,
        )


async def run_evaluation(
    rag,
    evaluation_dataset,
) -> list[dict[str, Any]]:

    results = []
    total = len(evaluation_dataset)

    for index, sample in enumerate(
        evaluation_dataset,
        start=1,
    ):
        print(
            f"[{index}/{total}] "
            f"{sample['question']}"
        )

        try:
            result = await rag.ainvoke(
                sample["question"]
            )
            documents = result.get(
                "documents",
                [],
            )
            results.append(
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "ground_truth": sample["ground_truth"],
                    "answer": result["answer"],
                    "contexts": serialize_documents(
                        documents
                    ),
                    "retrieval_calls": result.get(
                        "retrieval_calls",
                        0,
                    ),
                    "status": "success",
                    "error": None,
                }
            )

        except Exception as exc:

            print(
                f"ERROR on sample {sample['id']}: "
                f"{exc}"
            )

            results.append(
                {
                    "id": sample["id"],
                    "question": sample["question"],
                    "ground_truth": sample["ground_truth"],
                    "answer": None,
                    "contexts": [],
                    "retrieval_calls": 0,
                    "status": "error",
                    "error": str(exc),
                }
            )

    return results
