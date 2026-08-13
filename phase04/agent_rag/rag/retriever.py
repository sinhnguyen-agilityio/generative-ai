from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever as LangchainBM25Retriever


import datasets

class BM25Retriever:
    """Simple BM25-based retriever for document search."""
    
    def __init__(self, dataset_name="m-ric/huggingface_doc", default_k=3):
        self.default_k = default_k
        self.retriever = self._build_retriever(dataset_name)
    
    def _build_retriever(self, dataset_name: str) -> LangchainBM25Retriever:
        """Build a BM25 retriever from HuggingFace docs."""
        knowledge_base = datasets.load_dataset(dataset_name, split="train")
        
        # Create documents
        source_documents = [
            Document(
                page_content=row["text"],
                metadata={"source": row["source"].split("/")[1]},
            )
            for row in knowledge_base
        ]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )
        
        all_chunks = []
        for document in source_documents:
            chunks = text_splitter.split_documents([document])
            all_chunks.extend(chunks)
        
        # Simple deduplication
        unique_chunks = []
        seen_content = set()
        for chunk in all_chunks:
            if chunk.page_content not in seen_content:
                seen_content.add(chunk.page_content)
                unique_chunks.append(chunk)
        
        return LangchainBM25Retriever.from_documents(
            documents=unique_chunks,
            k=1,  # Will be overridden by retrieve method
        )
    
    def retrieve(self, query: str, top_k: int = None):
        """Retrieve documents for a given query."""
        if top_k is None:
            top_k = self.default_k
        self.retriever.k = top_k
        return self.retriever.invoke(query)
