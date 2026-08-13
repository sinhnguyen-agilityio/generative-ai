from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()

    def store_docs(self, collection: str, documents: list[Document]) -> Chroma:
        vector_store = Chroma(collection, self.embeddings)
        vector_store.add_documents(documents)
        return vector_store

    def similarity_search(self, query: str, collection: str, k: int = 4) -> list[Document]:
        vector_store = Chroma(collection, self.embeddings)
        results = vector_store.similarity_search(query, k=k)
        return results

    def retriever(self, collection: str):
        vector_store = Chroma(collection, self.embeddings)
        return vector_store.as_retriever()
