from langchain_core.documents import Document

from ingestion.loader import Loader
from ingestion.vector_store import VectorStore
from ingestion.chunker import Chunker


class IngestionPipeline:
    def __init__(self, loader: Loader, chunker: Chunker, vector_store: VectorStore):
        self.loader = loader
        self.chunker = chunker
        self.vector_store = vector_store

    def run(self, folder_path: str, collection_name: str) -> list[Document]:
        documents = self.loader.load_folder(folder_path)
        chunked_documents = self.chunker.chunk_documents(documents)
        # Store the chunked documents in the vector store
        self.vector_store.store_docs(collection_name, chunked_documents)

        return chunked_documents

    def similarity_search(self, query: str, collection_name: str, k: int = 4) -> list[Document]:
        # Perform a similarity search in the vector store
        results = self.vector_store.similarity_search(
            query, collection_name, k)
        return results
