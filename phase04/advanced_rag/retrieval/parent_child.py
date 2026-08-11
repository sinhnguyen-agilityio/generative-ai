
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from ingestion.chunker import Chunker

parent_splitter = Chunker(chunk_size=3000).text_splitter()
child_splitter = Chunker(chunk_size=500).text_splitter()

child_chunks_collection = Chroma(
    collection_name="uk_child_chunks",
    embedding_function=OpenAIEmbeddings(),
)

child_chunks_collection.reset_collection()
doc_store = InMemoryStore()

parent_doc_retriever = ParentDocumentRetriever(
    vectorstore=child_chunks_collection,
    docstore=doc_store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)
