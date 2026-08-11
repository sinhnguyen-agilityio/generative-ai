from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Chunker:
    def __init__(self, chunk_size=1000):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size)

    def chunk_documents(self, documents: list[Document]) -> list[Document]:
        chunked_results: list[Document] = []

        for doc in documents:
            # Split the document into chunks
            chunks = self.text_splitter.split_documents([doc])
            chunked_results.extend(chunks)

        return chunked_results
