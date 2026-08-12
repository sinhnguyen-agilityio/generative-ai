from langchain_core.vectorstores import VectorStoreRetriever
from .config import TOP_K
from .vectorstore import load_vectorstore


def get_retriever() -> VectorStoreRetriever:

    vectorstore = load_vectorstore()

    return vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K,
        }
    )
