from rag.dataset import (
    load_ragbench,
    build_evaluation_dataset,
    build_corpus,
)

from rag.ingestion import chunk_documents
from rag.vectorstore import create_vectorstore


def main():
    print("Loading RAGBench...")
    dataset = load_ragbench()

    print("Building evaluation dataset...")
    evaluation_dataset = build_evaluation_dataset(
        dataset,
        limit=10,
    )

    print("Building corpus...")
    corpus = build_corpus(
        evaluation_dataset
    )

    print("Chunking...")
    chunks = chunk_documents(corpus)

    print("Creating Chroma...")
    create_vectorstore(chunks)

    print("\nVector store built successfully.")
    print(f"Documents: {len(corpus)}")
    print(f"Chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
