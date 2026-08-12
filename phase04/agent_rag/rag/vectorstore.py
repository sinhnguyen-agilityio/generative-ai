from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from .config import (
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)


def get_embeddings():
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
    )


def create_vectorstore(documents):
    return Chroma.from_documents(
        documents=documents,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
    )


def load_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_PATH,
    )
