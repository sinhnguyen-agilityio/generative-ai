from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(
    documents: list[Document],
) -> list[Document]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(f"Documents: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    return chunks
