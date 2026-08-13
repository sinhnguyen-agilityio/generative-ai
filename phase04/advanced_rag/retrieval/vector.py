from ingestion.vector_store import VectorStore
from langchain_core.documents import Document


class VectorRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, collection: str, k: int = 4) -> list[Document]:
        return self.vector_store.similarity_search(query, collection, k)