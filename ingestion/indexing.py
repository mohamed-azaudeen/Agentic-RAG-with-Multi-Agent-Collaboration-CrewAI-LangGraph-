from ingestion.loader import load_documents
from ingestion.chunker import chunk_documents
from rag.embeddings import get_embeddings
from rag.vectorstore import create_vector_store

def build_index(data_path=r"C:\Users\azaru\Azar documents\Data Science Projects\Gen AI Projects\Agentic_RAG\local_rag_mcp_bot\data"):
    docs = load_documents(data_path)
    chunked_docs = chunk_documents(docs)
    embeddings = get_embeddings()
    db = create_vector_store(chunked_docs, embeddings)
    print("Index build and saved locally")


if __name__ == "__main__":
    build_index()
    