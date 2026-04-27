import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader

def load_documents(folder_path: str):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append(Document(page_content=content, metadata={"source": filename}))

        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = filename
                d.metadata["page"] = d.metadata.get("page", None)
            documents.extend(docs)

        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = filename
            documents.extend(docs)

        elif filename.endswith(".csv"):
            loader = CSVLoader(file_path)
            docs = loader.load()
            for d in docs:
                d.metadata["source"] = filename
            documents.extend(docs)


    return documents
