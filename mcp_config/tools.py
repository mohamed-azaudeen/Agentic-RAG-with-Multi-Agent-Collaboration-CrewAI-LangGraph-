from rag.retriever import load_retriever
from rag.embeddings import get_embeddings
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARIES_PATH = os.path.join(BASE_DIR, "summaries")

def direct_vector_search(query: str):
    """
    Standard Python function for high-speed retrieval.
    Bypasses LLM reasoning to prevent long wait times.
    """
    emb = get_embeddings()
    retriever = load_retriever(emb)
    docs = retriever.invoke(query)
    return [{"text": d.page_content, "metadata": d.metadata} for d in docs]


MCP_CONFIG = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", SUMMARIES_PATH],
        "env": {**os.environ}
    },
    "verification": "http://localhost:8001/python-repl"
}