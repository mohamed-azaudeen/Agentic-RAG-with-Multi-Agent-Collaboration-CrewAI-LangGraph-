import os
import logging
import time
import asyncio
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

from workflow.graph import workflow
from ingestion.loader import load_documents
from rag.vectorstore import create_vector_store
from mcp_config.tools import direct_vector_search


load_dotenv()
app = FastAPI()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s -%(levelname)s -%(message)s")

chat_history = []
vectorstore = None

class Question(BaseModel):
    query: str
    session_id: Optional[str] = "default_user"

@app.on_event("startup")
def startup_event():
    """Initializes the vectorstore once when the server starts."""
    global vectorstore
    logging.info("🚀 System starting up... initializing vectorstore.")
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if os.path.exists("vectorstore/index.faiss"):
            from langchain_community.vectorstores import FAISS
            vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
            logging.info("✅ Existing FAISS index loaded successfully.")
        else:
            docs = load_documents("data")
            vectorstore = create_vector_store(docs, embeddings)
            logging.info("✅ New FAISS index created from 'data' folder.")
    except Exception as e:
        logging.error(f"❌ Startup Error: {str(e)}")


@app.post("/ask")
async def ask_question(q: Question):
    """
    Invokes the multi-agent LangGraph workflow.
    """
    global chat_history
    if not q.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            lambda: workflow.invoke({"query": q.query, "history": chat_history})
        )
        logging.info(f"📩 Processing query: {q.query}")
        start_time = time.time()

        inputs = {
            "query": q.query, 
            "history": chat_history,
            "raw_docs": [] 
        }
        
        result = workflow.invoke(inputs)
        
        elapsed = time.time() - start_time
        
        chat_history = result.get("history", chat_history)
        
        answer = result.get("final_output", "The agents could not produce a verified answer.")
        
        logging.info(f"✅ Workflow complete in {elapsed:.2f}s")
        
        return {
            "answer": answer,
            "latency": f"{elapsed:.2f}s",
            "docs_retrieved": len(result.get("raw_docs", []))
        }

    except Exception as e:
        logging.error(f"❌ Error in /ask: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ingest")
def ingest_data(background_tasks: BackgroundTasks):
    """
    Re-scans the 'data' folder and updates the index.
    Running this as a BackgroundTask prevents the API from timing out.
    """
    def sync_ingestion():
        global vectorstore
        logging.info("Manual ingestion triggered...")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        docs = load_documents("data")
        vectorstore = create_vector_store(docs, embeddings)
        logging.info("Ingestion complete.")

    background_tasks.add_task(sync_ingestion)
    return {"status": "processing", "message": "Re-ingestion started in the background."}

@app.get("/health")
def health_check():
    """Checks if API keys and vectorstore are ready."""
    return {
        "status": "ready" if vectorstore else "initializing",
        "google_api": "configured" if os.getenv("GOOGLE_API_KEY") else "missing",
        "groq_api": "configured" if os.getenv("GROQ_API_KEY") else "missing"
    }