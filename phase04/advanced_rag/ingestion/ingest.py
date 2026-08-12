from langchain_chroma import Chroma
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_openai import OpenAIEmbeddings

from ingestion.chunker import Chunker
from ingestion.loader import Loader
from ingestion.sqlite_store import SQLiteDocStore


COLLECTION_NAME = "tourist_child_chunks"
CHROMA_PATH = "data/chroma"
DOCSTORE_PATH = "data/docstore.db"


def create_retriever(
    reingest: bool = False,
) -> ParentDocumentRetriever:
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    doc_store = SQLiteDocStore(
        DOCSTORE_PATH
    )
    parent_splitter = Chunker(
        chunk_size=3000
    ).text_splitter

    child_splitter = Chunker(
        chunk_size=500
    ).text_splitter

    # ----------------------------------------
    # Existing data
    # ----------------------------------------

    if (
        not reingest
        and vectorstore._collection.count() > 0
    ):
        print(
            "Using existing vector store: "
            f"{vectorstore._collection.count()} chunks"
        )

        return ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=doc_store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter,
            search_kwargs={"k": 4},
        )

    # ----------------------------------------
    # Re-ingestion
    # ----------------------------------------
    print("Starting re-ingestion...")
    vectorstore.reset_collection()

    # Clear old parent documents
    existing_keys = list(doc_store.yield_keys())

    if existing_keys:
        doc_store.mdelete(existing_keys)

    # ----------------------------------------
    # Load
    # ----------------------------------------
    loader = Loader()
    docs = loader.load_folder(
        "data/CilentoTouristInfo"
    )

    print(
        f"Loaded {len(docs)} source documents"
    )

    # ----------------------------------------
    # Retriever
    # ----------------------------------------
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=doc_store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 4},
    )

    # ----------------------------------------
    # Embed + persist
    # ----------------------------------------
    print("Embedding documents...")
    retriever.add_documents(docs)

    print(
        "Ingestion completed."
    )
    print(
        f"Child chunks: "
        f"{vectorstore._collection.count()}"
    )

    return retriever
