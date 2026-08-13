from langchain_core.documents import Document
from utils.pii import PIIMasker


class SecureContextProcessor:
    def __init__(self, pii_masker: PIIMasker):
        self.pii_masker = pii_masker

    def process(self, documents: list[Document]) -> list[Document]:
        secured_documents = []

        for document in documents:
            secured_documents.append(
                Document(
                    page_content=self.pii_masker.mask(
                        document.page_content
                    ),
                    metadata=document.metadata,
                )
            )

        return secured_documents
