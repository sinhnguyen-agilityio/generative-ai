from datasets import load_dataset
from langchain_core.documents import Document


def load_ragbench():
    return load_dataset(
        "rungalileo/ragbench",
        "hotpotqa",
    )


def build_evaluation_dataset(
    dataset,
    limit: int = 100,
) -> list[dict]:

    test_dataset = dataset["test"].select(
        range(min(limit, len(dataset["test"])))
    )

    return [
        {
            "id": row["id"],
            "question": row["question"],
            "ground_truth": row["response"],
            "reference_documents": row["documents"],
        }
        for row in test_dataset
    ]


def build_corpus(
    evaluation_dataset: list[dict],
) -> list[Document]:

    documents = []
    seen = set()

    for sample in evaluation_dataset:

        for index, text in enumerate(
            sample["reference_documents"]
        ):
            text = text.strip()

            if not text:
                continue

            if text in seen:
                continue

            seen.add(text)
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "dataset": "hotpotqa",
                        "source_question_id": sample["id"],
                        "document_index": index,
                    },
                )
            )

    print(f"Evaluation samples: {len(evaluation_dataset)}")
    print(f"Unique documents: {len(documents)}")

    return documents
