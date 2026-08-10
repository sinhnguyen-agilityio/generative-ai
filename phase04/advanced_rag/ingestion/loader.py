import os
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document


class Loader:
    def __init__(self):
        self.loader_classes = {
            'docx': Docx2txtLoader,
            'pdf': PyPDFLoader,
            'txt': TextLoader,
        }

    def _get_loader(self, filename) -> Docx2txtLoader | PyPDFLoader | TextLoader:
        _, file_extension = os.path.splitext(
            filename)  # A Extract the file extension
        # B Remove the leading dot from the extension
        file_extension = file_extension.lstrip('.')

        # C Get the loader class from the dictionary
        loader_class = self.loader_classes.get(file_extension)

        if loader_class:
            # D Instantiate and return the correct loader
            return loader_class(filename)
        else:
            raise ValueError(
                f"No loader available for file extension '{file_extension}'")

    def load_folder(self, folder_path) -> list[Document]:
        results: list[Document] = []

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    loader = self._get_loader(file_path)
                    documents = loader.load()  # Load the document
                    results.extend(documents)
                except ValueError as e:
                    print(e)
        return results
