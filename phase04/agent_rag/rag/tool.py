from langchain_core.tools import tool

from .retriever import get_retriever


class DocumentSearcher:

    def __init__(self):
        self.retriever = get_retriever()
        self.documents = []
        self.calls = 0

    def reset(self):
        self.documents = []
        self.calls = 0

    @property
    def tool(self):

        @tool
        def search_documents(query: str) -> str:
            """
            Search the knowledge base for information
            relevant to the user's question.
            """
            documents = self.retriever.invoke(
                query
            )
            self.calls += 1
            self.documents.extend(
                documents
            )

            if not documents:
                return (
                    "No relevant documents "
                    "were found."
                )

            return "\n\n".join(
                f"[Document {i + 1}]\n"
                f"{doc.page_content}"
                for i, doc in enumerate(documents)
            )

        return search_documents
